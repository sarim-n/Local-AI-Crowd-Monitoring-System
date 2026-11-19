"""
SMTP Notification engine for crowd monitoring incident alerts.
Formats HTML/plain text messages, attaches incident snapshots, measures delivery latency,
and logs delivery status.
"""

from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import os
from pathlib import Path
import smtplib
import time
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("EmailNotifier")


class EmailNotifier:
    """Manages SMTP alert dispatch and delivery logging."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        use_tls: Optional[bool] = None,
        recipients: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_SENDER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.use_tls = (
            use_tls
            if use_tls is not None
            else os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")
        )

        recipients_env = os.getenv("ALERT_RECIPIENTS", "")
        if recipients:
            self.recipients = recipients
        elif recipients_env:
            self.recipients = [r.strip() for r in recipients_env.split(",") if r.strip()]
        else:
            self.recipients = []

        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv("SMTP_ENABLED", "false").lower() in ("true", "1", "yes")

    def send_incident_alert(
        self,
        incident_id: str,
        zone_name: str,
        location: str,
        people_count: int,
        density_status: str,
        threshold_critical: int,
        sustained_seconds: float,
        timestamp_str: str,
        snapshot_path: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        """
        Send SMTP alert for a sustained overcrowding incident.

        Returns:
            Tuple of (success: bool, delivery_time_ms: float, message: str)
        """
        # If disabled (e.g. during development/verification), simulate without network transmission
        if not self.enabled:
            t0 = time.perf_counter()
            # Simulate message formatting and attachment parsing
            time.sleep(0.002)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            msg = (
                f"SMTP notifications disabled (SMTP_ENABLED=false). "
                f"Alert logged for incident {incident_id} ({len(self.recipients)} configured recipients)."
            )
            logger.info(msg)
            return False, round(elapsed_ms, 2), msg

        if not self.recipients:
            return False, 0.0, "No alert recipients configured in ALERT_RECIPIENTS."

        if not self.smtp_user or not self.smtp_password:
            return False, 0.0, "SMTP sender or password not configured in environment."

        t0 = time.perf_counter()
        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = f"🚨 [CRITICAL ALERT] Overcrowding Incident at {zone_name} (Count: {people_count})"
            msg["From"] = self.smtp_user
            msg["To"] = ", ".join(self.recipients)

            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="background-color: #dc3545; color: white; padding: 12px 20px; border-radius: 6px;">
                    <h2 style="margin: 0;">🚨 Sustained Overcrowding Incident Alert</h2>
                </div>
                <div style="padding: 15px 5px;">
                    <p>A sustained critical crowd density event has been detected by the local surveillance monitoring system.</p>
                    <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; font-weight: bold;">Incident ID:</td>
                            <td style="padding: 8px;">{incident_id}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; font-weight: bold;">Location:</td>
                            <td style="padding: 8px;">{location}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; font-weight: bold;">Monitoring Zone:</td>
                            <td style="padding: 8px;">{zone_name}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; font-weight: bold;">Current People Count:</td>
                            <td style="padding: 8px; font-size: 18px; color: #dc3545; font-weight: bold;">{people_count}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; font-weight: bold;">Critical Threshold:</td>
                            <td style="padding: 8px;">≥ {threshold_critical} people</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; font-weight: bold;">Sustained Duration:</td>
                            <td style="padding: 8px;">{sustained_seconds:.1f} seconds</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; font-weight: bold;">Detection Timestamp:</td>
                            <td style="padding: 8px;">{timestamp_str}</td>
                        </tr>
                    </table>
                    <p style="margin-top: 20px; font-size: 13px; color: #6c757d;">
                        An annotated snapshot of the surveillance frame has been attached to this email.
                    </p>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(html_body, "html"))

            # Attach snapshot if exists
            if snapshot_path and Path(snapshot_path).exists():
                try:
                    with open(snapshot_path, "rb") as f:
                        img_data = f.read()
                    image_part = MIMEImage(img_data, name=Path(snapshot_path).name)
                    msg.attach(image_part)
                except Exception as img_err:
                    logger.warning(f"Could not attach snapshot image: {img_err}")

            # Connect and send
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)

            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()

            delivery_time_ms = (time.perf_counter() - t0) * 1000.0
            success_msg = f"Alert email delivered to {len(self.recipients)} recipient(s) in {delivery_time_ms:.1f} ms."
            logger.info(success_msg)
            return True, round(delivery_time_ms, 2), success_msg

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = f"SMTP dispatch failed ({type(e).__name__}): {e}"
            logger.error(err_msg)
            return False, round(elapsed_ms, 2), err_msg

