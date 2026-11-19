"""
System resource monitoring module: CPU, RAM, and GPU utilization tracking.
"""

from dataclasses import dataclass
from typing import Optional
import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


@dataclass
class SystemMetrics:
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    ram_total_mb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_percent: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None

    def summary(self) -> str:
        s = f"CPU: {self.cpu_percent:.1f}% | RAM: {self.ram_percent:.1f}% ({self.ram_used_mb:.0f}/{self.ram_total_mb:.0f} MB)"
        if self.gpu_available and self.gpu_percent is not None:
            s += f" | GPU ({self.gpu_name}): {self.gpu_percent:.0f}% (VRAM: {self.gpu_mem_used_mb:.0f}/{self.gpu_mem_total_mb:.0f} MB)"
        return s


def collect_system_metrics() -> SystemMetrics:
    """Collect current hardware utilization snapshot."""
    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_pct = vm.percent
    ram_used = vm.used / (1024 * 1024)
    ram_total = vm.total / (1024 * 1024)

    gpu_avail = torch.cuda.is_available()
    gpu_name = None
    gpu_pct = None
    gpu_mem_used = None
    gpu_mem_total = None

    if gpu_avail:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            if HAS_NVML:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_pct = float(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = float(mem.used) / (1024 * 1024)
                gpu_mem_total = float(mem.total) / (1024 * 1024)
            else:
                # Fallback to PyTorch memory stats
                gpu_mem_used = torch.cuda.memory_allocated(0) / (1024 * 1024)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_pct = 0.0
        except Exception:
            pass

    return SystemMetrics(
        cpu_percent=round(cpu_pct, 1),
        ram_percent=round(ram_pct, 1),
        ram_used_mb=round(ram_used, 1),
        ram_total_mb=round(ram_total, 1),
        gpu_available=gpu_avail,
        gpu_name=gpu_name,
        gpu_percent=gpu_pct,
        gpu_mem_used_mb=gpu_mem_used,
        gpu_mem_total_mb=gpu_mem_total,
    )

