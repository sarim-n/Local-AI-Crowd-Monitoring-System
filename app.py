"""
Streamlit interactive monitoring dashboard for Local AI Crowd Monitoring (Phase 3).
Features:
- Instant video upload switching with automatic cache clearing (no page restart required)
- Duplicate and contained sub-box suppression (filters face/upper torso boxes inside full person boxes)
- Whole-frame crowd density assessment based on total bounding boxes per frame
- Sustained overcrowding incident detection, snapshot saving, and SMTP alerts
- Clean visual display without forced default zones (optional custom zones)
- Movement heatmaps, ByteTrack tracking, hardware utilization, and accuracy benchmarks
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.analytics.counting_evaluator import calculate_counting_accuracy, evaluate_dataset_samples
from src.analytics.system_metrics import collect_system_metrics
from src.monitoring.heatmap import CrowdHeatmap
from src.monitoring.tracker import CrowdDetectorTracker
from src.monitoring.video_stream import StreamType, VideoStreamSource
from src.monitoring.zone_manager import (
    STATUS_BG_HEX,
    STATUS_COLORS,
    DensityStatus,
    FrameCrowdMonitor,
    FrameCrowdThresholds,
    MonitoringZone,
    ZoneManager,
    ZoneThresholds,
)
from src.notification.email_notifier import EmailNotifier
from src.storage.database import DB_DIALECT, SessionLocal, init_db
from src.storage.repository import MonitoringRepository

# Page configuration
st.set_page_config(
    page_title="Local AI Crowd Monitoring System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for crowd cards, metrics, and alerts
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 14px;
        border-left: 5px solid #0E7AFE;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        color: white;
        font-weight: bold;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    .density-banner {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border: 1px solid #e9ecef;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .zone-card {
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background-color: #ffffff;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize stateful components for Streamlit."""
    init_db()

    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    if "frame_monitor" not in st.session_state:
        st.session_state.frame_monitor = FrameCrowdMonitor()

    if "zone_manager" not in st.session_state:
        db = SessionLocal()
        try:
            repo = MonitoringRepository(db)
            db_zones = repo.get_zones()
            zm = ZoneManager([])
            for dz in db_zones:
                zm.add_zone(
                    MonitoringZone(
                        id=dz.id,
                        name=dz.name,
                        x_min_pct=dz.x_min_pct,
                        y_min_pct=dz.y_min_pct,
                        x_max_pct=dz.x_max_pct,
                        y_max_pct=dz.y_max_pct,
                        thresholds=ZoneThresholds(
                            low=dz.low_threshold,
                            moderate=dz.moderate_threshold,
                            high=dz.high_threshold,
                            critical=dz.critical_threshold,
                            sustained_seconds=dz.sustained_seconds,
                        ),
                    )
                )
            st.session_state.zone_manager = zm
        finally:
            db.close()

    if "heatmap" not in st.session_state:
        st.session_state.heatmap = CrowdHeatmap()

    if "tracker" not in st.session_state:
        st.session_state.tracker = None

    if "current_weights_path" not in st.session_state:
        st.session_state.current_weights_path = ""

    if "temp_video_path" not in st.session_state:
        st.session_state.temp_video_path = None

    if "current_uploaded_file_id" not in st.session_state:
        st.session_state.current_uploaded_file_id = None

    if "email_notifier" not in st.session_state:
        st.session_state.email_notifier = EmailNotifier()


def get_or_load_tracker(weights_path: str, device: str) -> Optional[CrowdDetectorTracker]:
    """Cache or instantiate CrowdDetectorTracker with validation."""
    if (
        st.session_state.tracker is None
        or st.session_state.current_weights_path != weights_path
    ):
        try:
            tracker = CrowdDetectorTracker(weights_path=weights_path, device=device)
            st.session_state.tracker = tracker
            st.session_state.current_weights_path = weights_path
            return tracker
        except FileNotFoundError as fnf_err:
            st.sidebar.error(str(fnf_err))
            return None
        except Exception as ex:
            st.sidebar.error(f"Failed to load model: {ex}")
            return None
    return st.session_state.tracker


def main():
    init_session_state()

    st.title("👥 Local AI Crowd Monitoring Dashboard")
    st.caption(
        f"Frame-Level Crowd Density • Duplicate Box Suppression • ByteTrack • Sustained Overcrowding Alerts • "
        f"Database: {DB_DIALECT.upper()}"
    )

    # ========================== SIDEBAR CONTROLS ==========================
    st.sidebar.header("⚙️ Model Configuration")

    default_weights = "runs/detect/crowd_people_detector/weights/best.pt"
    weights_path_input = st.sidebar.text_input(
        "Model Checkpoint Path:",
        value=default_weights,
        help="Path to Phase 1 fine-tuned YOLO weights (.pt). Default: runs/detect/crowd_people_detector/weights/best.pt",
    )

    weights_path_obj = Path(weights_path_input)
    if weights_path_obj.exists():
        st.sidebar.success(f"✓ Model weights found ({weights_path_obj.stat().st_size / (1024*1024):.1f} MB)")
    else:
        st.sidebar.error(
            f"❌ File not found at '{weights_path_input}'.\n\n"
            f"**How to fix:**\n"
            f"1. Make sure Phase 1 training has completed: `python train.py`\n"
            f"2. Or enter the correct path to your trained `.pt` checkpoint.\n"
            f"*(The system will not silently switch to an arbitrary model)*"
        )

    device_option = st.sidebar.selectbox(
        "Execution Device:",
        options=["auto", "0", "cpu"],
        format_func=lambda x: "Auto (GPU if available)" if x == "auto" else ("CUDA:0 (GPU)" if x == "0" else "CPU"),
    )

    conf_thresh = st.sidebar.slider("Confidence Threshold:", 0.10, 0.90, 0.25, 0.05)
    iou_thresh = st.sidebar.slider("IoU / NMS Threshold:", 0.30, 0.90, 0.60, 0.05)
    enable_tracking = st.sidebar.checkbox("Enable ByteTrack Tracking", value=True)

    # ------------------ Duplicate Box Suppression ------------------
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Duplicate Box Filter")
    suppress_duplicates = st.sidebar.checkbox(
        "Filter Duplicate / Contained Boxes",
        value=True,
        help="Suppresses duplicate face or upper-torso boxes on the SAME person while preserving distinct individuals (e.g. standing behind, seated, or children) inside larger bounding boxes.",
    )
    containment_threshold = st.sidebar.slider(
        "Box Containment Threshold:",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="Containment ratio for same-person sub-box suppression. Distinct persons are automatically preserved.",
    )

    # ------------------ Frame-Level Crowd Density Thresholds ------------------
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Frame Crowd Thresholds")
    st.sidebar.caption("Evaluate crowd density based on total people detected in the frame:")

    col_th1, col_th2 = st.sidebar.columns(2)
    f_low = col_th1.number_input("Low / Normal ≤", min_value=1, max_value=100, value=5)
    f_mod = col_th2.number_input("Moderate ≤", min_value=f_low + 1, max_value=200, value=12)
    col_th3, col_th4 = st.sidebar.columns(2)
    f_high = col_th3.number_input("High ≤", min_value=f_mod + 1, max_value=300, value=18)
    f_crit = col_th4.number_input("Critical ≥", min_value=f_high + 1, max_value=500, value=22)
    f_sustained = st.sidebar.number_input(
        "Sustained Overcrowding (sec):",
        min_value=0.5,
        max_value=60.0,
        value=3.0,
        step=0.5,
        help="How long total count must remain ≥ Critical before an incident is logged and alerted.",
    )

    # Update frame monitor thresholds
    fm: FrameCrowdMonitor = st.session_state.frame_monitor
    fm.thresholds = FrameCrowdThresholds(
        low=f_low,
        moderate=f_mod,
        high=f_high,
        critical=f_crit,
        sustained_seconds=f_sustained,
    )

    # ------------------ Video Source Selection with Cache Invalidation ------------------
    st.sidebar.markdown("---")
    st.sidebar.header("📹 Video Source")

    source_type_label = st.sidebar.selectbox(
        "Select Input Source:",
        options=[
            StreamType.VIDEO_FILE.value,
            StreamType.WEBCAM.value,
            StreamType.RTSP.value,
            StreamType.DEMO.value,
        ],
    )
    source_type = StreamType(source_type_label)

    video_source_arg: Optional[str] = None
    cam_index = 0
    loop_video = True

    if source_type == StreamType.WEBCAM:
        cam_index = st.sidebar.number_input("Webcam Device Index:", min_value=0, max_value=5, value=0)
        st.sidebar.info("Uses local USB/integrated camera via OpenCV DirectShow.")

    elif source_type == StreamType.VIDEO_FILE:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Surveillance Video:",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload an MP4, AVI, or MOV video file for crowd analysis.",
        )
        loop_video = st.sidebar.checkbox("Loop Video", value=True)

        if uploaded_file is not None:
            # Generate unique signature for the uploaded file
            file_sig = f"{uploaded_file.name}_{uploaded_file.size}"

            # Detect if a DIFFERENT video was uploaded
            if st.session_state.current_uploaded_file_id != file_sig:
                # Stop any active stream immediately
                st.session_state.is_running = False

                # Prepare dedicated uploads directory
                upload_dir = Path("runs/temp_videos")
                upload_dir.mkdir(parents=True, exist_ok=True)
                new_video_path = str(upload_dir / f"upload_{int(time.time())}_{uploaded_file.name}")

                # Save new video file
                with open(new_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Clean up old temporary file if it existed
                old_temp = st.session_state.temp_video_path
                if old_temp and os.path.exists(old_temp) and old_temp != new_video_path:
                    try:
                        os.remove(old_temp)
                    except Exception:
                        pass

                # Update session state with new video
                st.session_state.temp_video_path = new_video_path
                st.session_state.current_uploaded_file_id = file_sig

                # Reset tracker session and heatmap for the new video
                st.session_state.heatmap.reset()
                if st.session_state.tracker:
                    st.session_state.tracker.reset_session()
                st.session_state.frame_monitor.reset()

                st.sidebar.success(f"✓ Loaded new video: {uploaded_file.name}")

            video_source_arg = st.session_state.temp_video_path

        else:
            # User cleared upload
            if st.session_state.current_uploaded_file_id is not None:
                st.session_state.is_running = False
                st.session_state.current_uploaded_file_id = None
                st.session_state.temp_video_path = None

            if Path("demo_crowd_surveillance.mp4").exists():
                video_source_arg = "demo_crowd_surveillance.mp4"
                st.sidebar.caption("Using default sample clip `demo_crowd_surveillance.mp4`.")

    elif source_type == StreamType.RTSP:
        rtsp_input = st.sidebar.text_input(
            "CCTV / RTSP Stream URL:",
            placeholder="rtsp://admin:password@192.168.1.100:554/h264",
            help="Enter a valid RTSP or HTTP camera stream address.",
        )
        video_source_arg = rtsp_input.strip() if rtsp_input else None

    elif source_type == StreamType.DEMO:
        st.sidebar.info("Streams real CCTV & surveillance test frames from `label-1/test/images` in a continuous loop.")

    # ------------------ Session Controls ------------------
    st.sidebar.markdown("---")
    st.sidebar.header("🎮 Session Controls")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_clicked = col_btn1.button("▶ Start", use_container_width=True, type="primary")
    stop_clicked = col_btn2.button("⏹ Stop", use_container_width=True)
    reset_clicked = st.sidebar.button("🔄 Reset Heatmap & Metrics", use_container_width=True)

    if start_clicked:
        st.session_state.is_running = True

    if stop_clicked:
        st.session_state.is_running = False

    if reset_clicked:
        st.session_state.heatmap.reset()
        st.session_state.frame_monitor.reset()
        st.session_state.zone_manager.reset_counts()
        if st.session_state.tracker:
            st.session_state.tracker.reset_session()
        st.sidebar.success("Session state, heatmap, and counts reset.")

    # ------------------ Display Options ------------------
    st.sidebar.markdown("---")
    st.sidebar.header("👁️ Display Options")
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
    show_ids = st.sidebar.checkbox("Show Tracking IDs", value=True)
    show_zones = st.sidebar.checkbox("Show Custom Zones (if configured)", value=True)
    overlay_heatmap = st.sidebar.checkbox("Overlay Heatmap on Video", value=False)
    heatmap_alpha = st.sidebar.slider("Heatmap Transparency:", 0.1, 0.9, 0.45, 0.05)

    # ------------------ Optional Custom Zones ------------------
    zm = st.session_state.zone_manager
    with st.sidebar.expander("📍 Optional Custom Zones (Advanced)", expanded=False):
        st.caption("You can define optional sub-zones if needed. By default, overall frame density is used.")
        new_zone_name = st.text_input("Zone Name:", value="Custom Sub-Zone")
        col_x1, col_x2 = st.columns(2)
        x_min_p = col_x1.slider("Left (X min %):", 0, 90, 10, 5) / 100.0
        x_max_p = col_x2.slider("Right (X max %):", 10, 100, 90, 5) / 100.0
        col_y1, col_y2 = st.columns(2)
        y_min_p = col_y1.slider("Top (Y min %):", 0, 90, 10, 5) / 100.0
        y_max_p = col_y2.slider("Bottom (Y max %):", 10, 100, 90, 5) / 100.0

        st.markdown("**Zone Thresholds:**")
        col_zt1, col_zt2 = st.columns(2)
        z_low = col_zt1.number_input("Zone Low ≤", min_value=1, max_value=50, value=2, key="z_th_low")
        z_mod = col_zt2.number_input("Zone Moderate ≤", min_value=z_low + 1, max_value=100, value=5, key="z_th_mod")
        col_zt3, col_zt4 = st.columns(2)
        z_high = col_zt3.number_input("Zone High ≤", min_value=z_mod + 1, max_value=150, value=8, key="z_th_high")
        z_crit = col_zt4.number_input("Zone Critical ≥", min_value=z_high + 1, max_value=200, value=10, key="z_th_crit")
        z_sust = st.number_input("Zone Sustained Overcrowding (s):", min_value=0.5, max_value=60.0, value=3.0, step=0.5, key="z_th_sust")

        if st.button("Add Custom Zone"):
            if x_min_p >= x_max_p or y_min_p >= y_max_p:
                st.error("Invalid coordinates: min must be strictly less than max.")
            else:
                new_id = f"zone_{int(time.time())}"
                new_thresholds = ZoneThresholds(
                    low=z_low,
                    moderate=z_mod,
                    high=z_high,
                    critical=z_crit,
                    sustained_seconds=z_sust,
                )
                new_zone = MonitoringZone(
                    id=new_id,
                    name=new_zone_name,
                    x_min_pct=x_min_p,
                    y_min_pct=y_min_p,
                    x_max_pct=x_max_p,
                    y_max_pct=y_max_p,
                    thresholds=new_thresholds,
                )
                zm.add_zone(new_zone)
                db = SessionLocal()
                try:
                    repo = MonitoringRepository(db)
                    repo.upsert_zone(
                        zone_id=new_id,
                        name=new_zone_name,
                        x_min_pct=x_min_p,
                        y_min_pct=y_min_p,
                        x_max_pct=x_max_p,
                        y_max_pct=y_max_p,
                        low_threshold=z_low,
                        moderate_threshold=z_mod,
                        high_threshold=z_high,
                        critical_threshold=z_crit,
                        sustained_seconds=z_sust,
                    )
                finally:
                    db.close()
                st.success(f"Added custom zone '{new_zone_name}' with critical threshold ≥ {z_crit}")

        if zm.zones:
            st.markdown("---")
            st.markdown("**Configured Zones:**")
            zones_to_remove = []
            for zid, zone in zm.zones.items():
                if st.button(f"🗑️ Delete {zone.name}", key=f"del_{zid}"):
                    zones_to_remove.append(zid)
            if zones_to_remove:
                db = SessionLocal()
                try:
                    repo = MonitoringRepository(db)
                    for zid in zones_to_remove:
                        zm.remove_zone(zid)
                        repo.delete_zone(zid)
                finally:
                    db.close()
                st.rerun()

    # ========================== TOP KPI ROW ==========================
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    metric_people = kpi_col1.metric("👥 Current People in Frame", "0")
    metric_status = kpi_col2.metric("🚦 Crowd Density Status", "LOW")
    metric_unique = kpi_col3.metric("🏷️ Unique Tracked IDs", "0")
    metric_incidents = kpi_col4.metric("🚨 Total Incidents", "0")
    metric_fps = kpi_col5.metric("⚡ Processing Speed", "0.0 FPS")

    # ========================== MAIN LAYOUT ==========================
    main_col1, main_col2 = st.columns([3, 2])

    with main_col1:
        st.subheader("🎥 Live Surveillance Stream")
        crowd_banner_placeholder = st.empty()
        video_placeholder = st.empty()
        status_banner = st.empty()

    with main_col2:
        st.subheader("📊 Analytics & Controls")
        tab_density, tab_incidents, tab_heatmap, tab_accuracy, tab_system = st.tabs(
            [
                "👥 Crowd Density",
                "🚨 Incidents & Alerts",
                "🔥 Heatmap",
                "📈 Accuracy Benchmark",
                "💻 System Hardware",
            ]
        )

        with tab_density:
            density_cards_placeholder = st.empty()

        with tab_incidents:
            incidents_placeholder = st.empty()

        with tab_heatmap:
            heatmap_placeholder = st.empty()

        with tab_accuracy:
            st.markdown("#### Ground-Truth People Counting Benchmark")
            st.caption(
                "Benchmarks detection counting accuracy against ground-truth annotations from the held-out test dataset."
            )
            col_b1, col_b2 = st.columns([2, 1])
            bench_samples = col_b1.slider("Test Samples to Benchmark:", 5, 50, 20, 5)
            run_bench_btn = col_b2.button("Run Accuracy Benchmark", type="secondary")
            bench_results_placeholder = st.empty()

            st.markdown("---")
            st.markdown("#### Phase 1 Held-Out Test Evaluation Metrics")
            st.caption("Evaluated on 109 test images (585 people instances) on NVIDIA RTX 2050 GPU:")
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Precision", "77.61%")
            col_p2.metric("Recall", "73.45%")
            col_p3.metric("mAP@0.50", "72.60%")
            col_p4.metric("mAP@0.50:0.95", "40.52%")

            if run_bench_btn:
                with st.spinner("Benchmarking counting accuracy on test set..."):
                    try:
                        bench_res = evaluate_dataset_samples(
                            weights_path=weights_path_input,
                            test_dir="label-1/test",
                            num_samples=bench_samples,
                            conf_threshold=conf_thresh,
                            device=device_option if device_option != "auto" else None,
                        )
                        with bench_results_placeholder.container():
                            st.success(f"Benchmarked {bench_res.num_samples} frames:")
                            b_m1, b_m2, b_m3 = st.columns(3)
                            b_m1.metric("Overall Accuracy", f"{bench_res.overall_accuracy_pct:.1f}%")
                            b_m2.metric("Mean Error (MAE)", f"{bench_res.mean_absolute_error:.2f} people")
                            b_m3.metric("RMSE", f"{bench_res.root_mean_squared_error:.2f}")
                            st.dataframe(pd.DataFrame(bench_res.sample_details).head(10), use_container_width=True)
                    except Exception as b_err:
                        bench_results_placeholder.error(f"Benchmark error: {b_err}")

        with tab_system:
            sys_metrics_placeholder = st.empty()

    # Pre-flight Model Validation
    tracker = get_or_load_tracker(weights_path_input, device_option)
    if tracker is None:
        status_banner.error(
            "⚠️ Monitoring cannot start: Model weights are missing or invalid. Check the path in the sidebar."
        )
        return

    # Render initial system metrics
    sys_snapshot = collect_system_metrics()
    with sys_metrics_placeholder.container():
        st.markdown(f"**Database:** `{DB_DIALECT.upper()}`")
        s_c1, s_c2 = st.columns(2)
        s_c1.metric("CPU Utilization", f"{sys_snapshot.cpu_percent:.1f}%")
        s_c2.metric("RAM Utilization", f"{sys_snapshot.ram_percent:.1f}%")
        if sys_snapshot.gpu_available:
            g_c1, g_c2 = st.columns(2)
            g_c1.metric("GPU Utilization", f"{sys_snapshot.gpu_percent:.0f}%")
            g_c2.metric("GPU VRAM Used", f"{sys_snapshot.gpu_mem_used_mb:.0f} MB / {sys_snapshot.gpu_mem_total_mb:.0f} MB")

    # Render existing incidents from DB
    db = SessionLocal()
    try:
        repo = MonitoringRepository(db)
        recent_incs = repo.get_incidents(limit=10)
        with incidents_placeholder.container():
            if recent_incs:
                inc_data = [
                    {
                        "ID": inc.id,
                        "Zone/Location": inc.zone_name,
                        "Count": inc.people_count,
                        "Status": inc.density_status,
                        "Time (UTC)": inc.started_at.strftime("%H:%M:%S"),
                        "Email Alert": inc.email_status,
                    }
                    for inc in recent_incs
                ]
                st.dataframe(pd.DataFrame(inc_data), use_container_width=True)
                if recent_incs[0].snapshot_path and os.path.exists(recent_incs[0].snapshot_path):
                    st.caption(f"📸 Latest Incident Snapshot: `{recent_incs[0].id}`")
                    st.image(recent_incs[0].snapshot_path, use_container_width=True)
            else:
                st.info("No sustained overcrowding incidents logged yet.")
    finally:
        db.close()

    # Initial static crowd banner
    crowd_banner_placeholder.markdown(
        f"""
        <div class="density-banner">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 16px; font-weight: bold;">Scene Crowd Density:</span>
                <span class="status-badge" style="background-color: {STATUS_BG_HEX[fm.status]};">
                    {fm.status.value} ({fm.current_count} people)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if user clicked start
    if not st.session_state.is_running:
        status_banner.info(
            "ℹ️ Monitoring is paused. Select your video source in the sidebar and click **'▶ Start'** to begin real-time analysis."
        )
        if source_type == StreamType.DEMO:
            test_img_path = Path("label-1/test/images")
            if test_img_path.exists():
                first_img = next(test_img_path.glob("*.jpg"), None)
                if first_img:
                    preview_bgr = cv2.imread(str(first_img))
                    if preview_bgr is not None:
                        if show_zones and zm.zones:
                            zm.draw_zones(preview_bgr)
                        video_placeholder.image(
                            cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB),
                            caption="Static Preview (Ready to Start)",
                            use_container_width=True,
                        )
        return

    # ------------------ Open Video Stream ------------------
    stream = VideoStreamSource(
        stream_type=source_type,
        source=video_source_arg,
        cam_index=cam_index,
        loop=loop_video,
    )

    open_success, open_msg = stream.open()
    if not open_success:
        st.session_state.is_running = False
        status_banner.error(f"❌ Stream Connection Failed:\n\n{open_msg}")
        return

    status_banner.success(f"✓ {open_msg}")

    # ========================== PROCESSING LOOP ==========================
    heatmap = st.session_state.heatmap
    notifier: EmailNotifier = st.session_state.email_notifier
    prev_time = time.time()
    fps_history = []
    frame_counter = 0

    snapshot_dir = Path(os.getenv("INCIDENT_SNAPSHOT_DIR", "runs/incidents")).resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    location_name = os.getenv("MONITORING_LOCATION", "Main Surveillance Scene")

    try:
        while st.session_state.is_running:
            read_success, frame, is_eof = stream.read_frame()

            if is_eof or not read_success or frame is None:
                if is_eof:
                    status_banner.info("⏹ End of video stream reached.")
                else:
                    status_banner.warning("⚠️ Stream disconnected or frame dropped.")
                st.session_state.is_running = False
                break

            frame_counter += 1
            now_ts = time.time()
            h, w = frame.shape[:2]

            # 1. Detection and Tracking with Duplicate Box Suppression
            detections, annotated_frame = tracker.process_frame(
                frame=frame,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                enable_tracking=enable_tracking,
                draw_annotations=show_boxes,
                show_track_id=show_ids,
                suppress_duplicates=suppress_duplicates,
                containment_threshold=containment_threshold,
            )

            total_people_count = len(detections)
            footprints = [d.bottom_center for d in detections]

            # 2. Evaluate Total Bounding Boxes Frame Crowd Density & Sustained Overcrowding
            density_status, is_frame_incident, duration_crit = fm.update(total_people_count, now=now_ts)

            # 3. Handle Sustained Overcrowding Incident on Total Frame Count
            if is_frame_incident:
                timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                inc_id = f"INC-FRAME-{int(now_ts)}"
                snapshot_filename = f"{inc_id}.jpg"
                snapshot_full_path = str(snapshot_dir / snapshot_filename)
                cv2.imwrite(snapshot_full_path, annotated_frame)

                # Send / Log Alert
                success, delivery_ms, alert_msg = notifier.send_incident_alert(
                    incident_id=inc_id,
                    zone_name="Entire Surveillance Scene",
                    location=location_name,
                    people_count=total_people_count,
                    density_status=density_status.value,
                    threshold_critical=fm.thresholds.critical,
                    sustained_seconds=fm.thresholds.sustained_seconds,
                    timestamp_str=timestamp_str,
                    snapshot_path=snapshot_full_path,
                )

                email_status = "SENT" if success else ("DISABLED" if not notifier.enabled else "FAILED")

                # Persist Incident to DB
                db = SessionLocal()
                try:
                    repo = MonitoringRepository(db)
                    repo.create_incident(
                        incident_id=inc_id,
                        zone_id="entire_frame",
                        zone_name="Entire Surveillance Scene",
                        location=location_name,
                        people_count=total_people_count,
                        density_status=density_status.value,
                        threshold_critical=fm.thresholds.critical,
                        sustained_seconds=fm.thresholds.sustained_seconds,
                        snapshot_path=snapshot_full_path,
                        email_status=email_status,
                        email_error=None if success else alert_msg,
                    )
                    for r in notifier.recipients or ["admin@example.com"]:
                        repo.log_notification(
                            incident_id=inc_id,
                            recipient=r,
                            status=email_status,
                            delivery_time_ms=delivery_ms,
                            error_message=None if success else alert_msg,
                        )
                finally:
                    db.close()

            # 4. Update optional custom zones (if configured)
            if zm.zones:
                zm.update_detections(footprints, frame_w=w, frame_h=h)
                triggered_zones = zm.evaluate_incidents(now_ts)
                for tz, t_dur in triggered_zones:
                    z_inc_id = f"INC-{tz.id}-{int(now_ts)}"
                    tz.active_incident_id = z_inc_id
                    z_snap_name = f"{z_inc_id}.jpg"
                    z_snap_full = str(incidents_dir / z_snap_name)
                    cv2.imwrite(z_snap_full, annotated_frame)

                    z_success, z_del_ms, z_msg = notifier.send_incident_alert(
                        incident_id=z_inc_id,
                        zone_name=tz.name,
                        location=location_name,
                        people_count=tz.current_count,
                        density_status=tz.status.value,
                        threshold_critical=tz.thresholds.critical,
                        sustained_seconds=tz.thresholds.sustained_seconds,
                        timestamp_str=timestamp_str,
                        snapshot_path=z_snap_full,
                    )
                    z_email_status = "SENT" if z_success else ("DISABLED" if not notifier.enabled else "FAILED")

                    db = SessionLocal()
                    try:
                        repo = MonitoringRepository(db)
                        repo.create_incident(
                            incident_id=z_inc_id,
                            zone_id=tz.id,
                            zone_name=tz.name,
                            location=location_name,
                            people_count=tz.current_count,
                            density_status=tz.status.value,
                            threshold_critical=tz.thresholds.critical,
                            sustained_seconds=tz.thresholds.sustained_seconds,
                            snapshot_path=z_snap_full,
                            email_status=z_email_status,
                            email_error=None if z_success else z_msg,
                        )
                        for r in notifier.recipients or ["admin@example.com"]:
                            repo.log_notification(
                                incident_id=z_inc_id,
                                recipient=r,
                                status=z_email_status,
                                delivery_time_ms=z_del_ms,
                                error_message=None if z_success else z_msg,
                            )
                    finally:
                        db.close()

                if show_zones:
                    zm.draw_zones(annotated_frame)

            # 5. Heatmap Accumulation & Optional Overlay
            heatmap.update(footprints, frame_w=w, frame_h=h)
            if overlay_heatmap:
                annotated_frame = heatmap.blend_overlay(annotated_frame, alpha=heatmap_alpha)

            # 6. FPS Calculation
            dt = time.time() - prev_time
            prev_time = time.time()
            if dt > 0:
                fps = 1.0 / dt
                fps_history.append(fps)
                if len(fps_history) > 20:
                    fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            unique_ids = len(tracker.unique_track_ids)
            total_incidents = fm.total_incidents_triggered + sum(z.total_incidents_triggered for z in zm.zones.values())

            # Top KPI Updates
            metric_people.metric("👥 Current People in Frame", f"{total_people_count}")
            metric_status.metric("🚦 Crowd Density Status", f"{density_status.value}")
            metric_unique.metric("🏷️ Unique Tracked IDs", f"{unique_ids}")
            metric_incidents.metric("🚨 Total Incidents", f"{total_incidents}")
            metric_fps.metric("⚡ Processing Speed", f"{avg_fps:.1f} FPS")

            # 7. Render Crowd Density Banner
            color_hex = STATUS_BG_HEX[density_status]
            overcrowd_alert_html = ""
            if fm.overcrowded_since is not None and total_people_count >= fm.thresholds.critical:
                elapsed = now_ts - fm.overcrowded_since
                if fm.in_active_incident:
                    overcrowd_alert_html = (
                        f"<div style='color: #dc3545; font-size: 13px; font-weight: bold; margin-top: 6px;'>"
                        f"🚨 SUSTAINED CRITICAL OVERCROWDING INCIDENT ACTIVE ({elapsed:.1f}s)</div>"
                    )
                else:
                    overcrowd_alert_html = (
                        f"<div style='color: #fd7e14; font-size: 13px; margin-top: 6px;'>"
                        f"⏱️ Overcrowding: {elapsed:.1f}s / {fm.thresholds.sustained_seconds:.0f}s threshold before alert</div>"
                    )

            crowd_banner_placeholder.markdown(
                f"""
                <div class="density-banner" style="border-left: 6px solid {color_hex};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 17px; font-weight: bold;">Crowd Density:</span>
                            <span style="font-size: 16px; margin-left: 8px;">{total_people_count} people in frame</span>
                        </div>
                        <span class="status-badge" style="background-color: {color_hex}; font-size: 14px;">
                            {density_status.value}
                        </span>
                    </div>
                    {overcrowd_alert_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 8. Render Video Frame
            video_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            # 9. Render Density Details (Tab 1)
            with density_cards_placeholder.container():
                st.markdown("#### Frame-Level Density Thresholds")
                st.markdown(
                    f"""
                    <div class="zone-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>Total Frame People Count: {total_people_count}</strong>
                            <span class="status-badge" style="background-color: {color_hex};">
                                {density_status.value}
                            </span>
                        </div>
                        <div style="font-size: 13px; color: #495057; margin-top: 6px;">
                            🟢 Low ≤ {fm.thresholds.low} | 🟡 Mod ≤ {fm.thresholds.moderate} | 🟠 High ≤ {fm.thresholds.high} | 🔴 Crit ≥ {fm.thresholds.critical}
                        </div>
                        <div style="font-size: 12px; color: #6c757d; margin-top: 4px;">
                            Sustained duration required for incident: {fm.thresholds.sustained_seconds:.1f} seconds
                        </div>
                        {overcrowd_alert_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if zm.zones:
                    st.markdown("#### Optional Custom Sub-Zones")
                    for zid, z in zm.zones.items():
                        z_color = STATUS_BG_HEX.get(z.status, "#6c757d")
                        st.markdown(
                            f"""
                            <div class="zone-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong>{z.name}</strong>
                                    <span class="status-badge" style="background-color: {z_color};">
                                        {z.status.value} ({z.current_count})
                                    </span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # 10. Render Incidents Table (Tab 2)
            if frame_counter % 15 == 0:
                db = SessionLocal()
                try:
                    repo = MonitoringRepository(db)
                    recent_incs = repo.get_incidents(limit=8)
                    with incidents_placeholder.container():
                        if recent_incs:
                            inc_data = [
                                {
                                    "ID": inc.id,
                                    "Location": inc.zone_name,
                                    "Count": inc.people_count,
                                    "Time (UTC)": inc.started_at.strftime("%H:%M:%S"),
                                    "Email Status": inc.email_status,
                                }
                                for inc in recent_incs
                            ]
                            st.dataframe(pd.DataFrame(inc_data), use_container_width=True)
                            if recent_incs[0].snapshot_path and os.path.exists(recent_incs[0].snapshot_path):
                                st.caption(f"📸 Latest Incident Snapshot: `{recent_incs[0].id}`")
                                st.image(recent_incs[0].snapshot_path, use_container_width=True)
                        else:
                            st.info("No sustained overcrowding incidents logged yet.")
                finally:
                    db.close()

            # 11. Render Standalone Heatmap (Tab 3)
            if heatmap.total_points_accumulated > 0 and frame_counter % 20 == 0:
                color_heat = heatmap.get_color_heatmap(frame_w=w, frame_h=h)
                heatmap_placeholder.image(
                    cv2.cvtColor(color_heat, cv2.COLOR_BGR2RGB),
                    caption=f"Cumulative Movement Heatmap ({heatmap.total_points_accumulated} footprint positions)",
                    use_container_width=True,
                )

            # 12. Periodic Hardware Snapshot (Tab 5)
            if frame_counter % 15 == 0:
                sys_snap = collect_system_metrics()
                with sys_metrics_placeholder.container():
                    st.markdown(f"**Database Backend:** `{DB_DIALECT.upper()}`")
                    c1, c2 = st.columns(2)
                    c1.metric("CPU Utilization", f"{sys_snap.cpu_percent:.1f}%")
                    c2.metric("RAM Utilization", f"{sys_snap.ram_percent:.1f}%")
                    if sys_snap.gpu_available:
                        g1, g2 = st.columns(2)
                        g1.metric("GPU Utilization", f"{sys_snap.gpu_percent:.0f}%")
                        g2.metric("GPU VRAM Used", f"{sys_snap.gpu_mem_used_mb:.0f} MB / {sys_snap.gpu_mem_total_mb:.0f} MB")

            time.sleep(0.01)

    except Exception as run_err:
        status_banner.error(f"⚠️ Error during stream processing: {run_err}")
    finally:
        stream.release()


if __name__ == "__main__":
    main()
