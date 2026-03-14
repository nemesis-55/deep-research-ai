"""
System Metrics — Deep Research AI (Apple Silicon / Mac Mini M4)

Collects hardware utilisation metrics without sudo:
  • CPU  — overall % + per-core, frequency, load average
  • RAM  — total / used / available / percent, swap
  • GPU  — utilisation %, VRAM in-use / allocated  (via ioreg, no sudo)
  • ANE  — reported when ioreg exposes it
  • Disk — total / free / percent for the uploads volume
  • Process — RSS of this uvicorn worker, tokens/sec (if tracked)

ioreg path: IOAccelerator → PerformanceStatistics
  Renderer Utilization %  → GPU shader/compute load
  Device Utilization %    → overall GPU engine
  Tiler Utilization %     → geometry/vertex load
  In use system memory    → active VRAM bytes (unified memory slice)
  Alloc system memory     → total VRAM allocated bytes

Metrics logging:
  Writes a JSONL file at <log_dir>/metrics_<date>.jsonl.
  Each line is a JSON object with timestamp + full snapshot.
  Rotates daily — one file per day.  Keeps last N days.
  Inference events (chat/research completions) are written inline
  alongside the periodic hardware snapshots.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── ioreg GPU sampling ────────────────────────────────────────────────────────

_IOREG_FIELDS = {
    "Device Utilization %":      "gpu_util_pct",
    "Renderer Utilization %":    "gpu_renderer_pct",
    "Tiler Utilization %":       "gpu_tiler_pct",
    "In use system memory":      "gpu_vram_used_bytes",
    "Alloc system memory":       "gpu_vram_alloc_bytes",
}


def _read_ioreg_gpu() -> Dict:
    """
    Parse Apple Silicon GPU stats from ioreg.
    Returns an empty dict if ioreg is unavailable or parsing fails.
    Cost: ~15 ms subprocess fork on M-series.
    """
    try:
        r = subprocess.run(
            ["ioreg", "-r", "-d1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=3,
        )
        raw = r.stdout
    except Exception as e:
        logger.debug(f"[Metrics] ioreg failed: {e}")
        return {}

    stats_match = re.search(r'"PerformanceStatistics"\s*=\s*\{([^}]+)\}', raw, re.DOTALL)
    if not stats_match:
        return {}

    block = stats_match.group(1)
    out: Dict = {}
    for label, key in _IOREG_FIELDS.items():
        m = re.search(rf'"{re.escape(label)}"\s*=\s*(\d+)', block)
        if m:
            out[key] = int(m.group(1))

    return out


# ── Main collection function ──────────────────────────────────────────────────

def collect() -> Dict:
    """
    Collect a full snapshot of system metrics.

    Returns a dict safe to JSON-serialise directly into the /metrics response.
    All sizes are in bytes unless the key ends in _pct, _mhz, _gb, _mb.
    """
    import psutil

    ts = time.time()

    # ── CPU ──────────────────────────────────────────────────────────────────
    cpu_pct      = psutil.cpu_percent(interval=None)   # non-blocking; caller must
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)  # have called once before
    freq         = psutil.cpu_freq()
    load1, load5, load15 = psutil.getloadavg()
    cpu_count_logical  = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)

    cpu = {
        "pct":            round(cpu_pct, 1),
        "per_core":       [round(c, 1) for c in cpu_per_core],
        "freq_mhz":       round(freq.current * 1000, 0) if freq else None,  # psutil returns GHz on Apple
        "freq_max_mhz":   round(freq.max    * 1000, 0) if freq else None,
        "load1":          round(load1,  2),
        "load5":          round(load5,  2),
        "load15":         round(load15, 2),
        "cores_logical":  cpu_count_logical,
        "cores_physical": cpu_count_physical,
    }

    # ── RAM ──────────────────────────────────────────────────────────────────
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()

    ram = {
        "total_bytes":     vm.total,
        "used_bytes":      vm.used,
        "available_bytes": vm.available,
        "pct":             round(vm.percent, 1),
        "swap_total_bytes": swap.total,
        "swap_used_bytes":  swap.used,
        "swap_pct":         round(swap.percent, 1),
    }

    # ── GPU (Apple Silicon unified memory via ioreg) ──────────────────────────
    gpu_raw = _read_ioreg_gpu()
    gpu = {
        "util_pct":        gpu_raw.get("gpu_util_pct"),
        "renderer_pct":    gpu_raw.get("gpu_renderer_pct"),
        "tiler_pct":       gpu_raw.get("gpu_tiler_pct"),
        "vram_used_bytes": gpu_raw.get("gpu_vram_used_bytes"),
        "vram_alloc_bytes":gpu_raw.get("gpu_vram_alloc_bytes"),
        # Convenience: % of total RAM consumed by GPU
        "vram_used_pct":   round(
            gpu_raw["gpu_vram_used_bytes"] / vm.total * 100, 1
        ) if gpu_raw.get("gpu_vram_used_bytes") else None,
    }

    # ── Disk ─────────────────────────────────────────────────────────────────
    try:
        disk_root = psutil.disk_usage("/")
        disk = {
            "total_bytes": disk_root.total,
            "used_bytes":  disk_root.used,
            "free_bytes":  disk_root.free,
            "pct":         round(disk_root.percent, 1),
        }
    except Exception:
        disk = {}

    # External T7 Shield (optional)
    ext_disk: Optional[Dict] = None
    try:
        d = psutil.disk_usage("/Volumes/T7 Shield")
        ext_disk = {
            "total_bytes": d.total,
            "used_bytes":  d.used,
            "free_bytes":  d.free,
            "pct":         round(d.percent, 1),
        }
    except Exception:
        pass

    # ── This process ─────────────────────────────────────────────────────────
    try:
        proc = psutil.Process()
        proc_cpu  = round(proc.cpu_percent(interval=None), 1)
        proc_rss  = proc.memory_info().rss
        proc_vms  = proc.memory_info().vms
        proc_threads = proc.num_threads()
    except Exception:
        proc_cpu = proc_rss = proc_vms = proc_threads = None

    process = {
        "cpu_pct":     proc_cpu,
        "rss_bytes":   proc_rss,
        "vms_bytes":   proc_vms,
        "threads":     proc_threads,
    }

    return {
        "timestamp": ts,
        "cpu":       cpu,
        "ram":       ram,
        "gpu":       gpu,
        "disk":      disk,
        "ext_disk":  ext_disk,
        "process":   process,
    }


# ── Warm-up: psutil cpu_percent needs one prior call to calibrate ─────────────

def _warmup() -> None:
    try:
        import psutil
        psutil.cpu_percent(interval=0.1)
        p = psutil.Process()
        p.cpu_percent(interval=None)
    except Exception:
        pass

_warmup()


# ── Metrics Logger ────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Background thread that samples hardware metrics every `interval_s` seconds
    and appends JSONL records to a rotating daily log file.

    Also exposes `log_inference()` for on-demand inference event logging.

    Usage:
        ml = MetricsLogger(log_dir=Path("logs/metrics"), interval_s=30)
        ml.start()
        ...
        ml.log_inference(role="chat", prompt_tokens=42, output_tokens=128,
                         elapsed_s=3.2, complexity="technical")
        ...
        ml.stop()
    """

    def __init__(
        self,
        log_dir:    Path  = Path("logs/metrics"),
        interval_s: float = 30.0,
        keep_days:  int   = 7,
    ) -> None:
        self.log_dir    = Path(log_dir)
        self.interval_s = interval_s
        self.keep_days  = keep_days
        self._stop      = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock      = threading.Lock()
        # Session-level counters
        self._session_requests: int   = 0
        self._session_tokens:   int   = 0
        self._session_start:    float = time.time()

    # ── File management ───────────────────────────────────────────────────────

    def _today_path(self) -> Path:
        """Return the path for today's JSONL file, creating dirs as needed."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        name = f"metrics_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        return self.log_dir / name

    def _rotate(self) -> None:
        """Delete log files older than keep_days."""
        cutoff = time.time() - self.keep_days * 86_400
        for f in self.log_dir.glob("metrics_*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.debug(f"[MetricsLog] Rotated old log: {f.name}")
            except Exception:
                pass

    # ── Write ─────────────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        """Append one JSON line to today's log file (thread-safe)."""
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        line = json.dumps(record, separators=(",", ":"))
        path = self._today_path()
        with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception as e:
                logger.warning(f"[MetricsLog] Write failed: {e}")

    # ── Hardware snapshot ─────────────────────────────────────────────────────

    def _sample_and_log(self) -> None:
        """Collect a hardware snapshot and write it."""
        try:
            snap = collect()
            # Summarise to keep log compact (no per-core array)
            record = {
                "type": "hw",
                "cpu_pct":          snap["cpu"]["pct"],
                "cpu_load1":        snap["cpu"]["load1"],
                "ram_pct":          snap["ram"]["pct"],
                "ram_used_gb":      round(snap["ram"]["used_bytes"] / 1e9, 2),
                "gpu_pct":          snap["gpu"].get("util_pct"),
                "gpu_renderer_pct": snap["gpu"].get("renderer_pct"),
                "vram_used_gb":     round(snap["gpu"]["vram_used_bytes"] / 1e9, 2)
                                    if snap["gpu"].get("vram_used_bytes") else None,
                "proc_rss_gb":      round(snap["process"]["rss_bytes"] / 1e9, 2)
                                    if snap["process"].get("rss_bytes") else None,
                "proc_cpu_pct":     snap["process"].get("cpu_pct"),
                "session_requests": self._session_requests,
                "session_tokens":   self._session_tokens,
                "session_uptime_s": round(time.time() - self._session_start, 0),
            }
            self._write(record)
        except Exception as e:
            logger.debug(f"[MetricsLog] Sample error: {e}")

    # ── Public: inference event ───────────────────────────────────────────────

    def log_inference(
        self,
        role:          str,
        prompt_chars:  int,
        output_chars:  int,
        output_tokens: int,
        elapsed_s:     float,
        complexity:    str = "",
        model_name:    str = "",
        runtime:       str = "",
    ) -> None:
        """
        Log one inference completion event.
        Called from model_manager.generate() after every successful generation.
        """
        self._session_requests += 1
        self._session_tokens   += output_tokens
        tok_per_s = round(output_tokens / elapsed_s, 1) if elapsed_s > 0 else 0
        record = {
            "type":          "inference",
            "role":          role,
            "complexity":    complexity,
            "model":         model_name,
            "runtime":       runtime,
            "prompt_chars":  prompt_chars,
            "output_chars":  output_chars,
            "output_tokens": output_tokens,
            "elapsed_s":     round(elapsed_s, 3),
            "tok_per_s":     tok_per_s,
            "session_req_n": self._session_requests,
        }
        self._write(record)
        logger.info(
            f"[Metrics] {role} | {complexity or 'n/a'} | "
            f"{output_tokens} tok | {elapsed_s:.2f}s | {tok_per_s} tok/s | "
            f"{model_name}"
        )

    def log_event(self, event_type: str, **kwargs) -> None:
        """Log an arbitrary named event (startup, shutdown, error, etc.)."""
        record = {"type": event_type, **kwargs}
        self._write(record)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background sampling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="metrics-logger"
        )
        self._thread.start()
        logger.info(
            f"[MetricsLog] Started — interval={self.interval_s}s  "
            f"dir={self.log_dir}  keep={self.keep_days}d"
        )
        self.log_event("startup", interval_s=self.interval_s)

    def stop(self) -> None:
        """Stop the background thread gracefully."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.log_event("shutdown",
                        session_requests=self._session_requests,
                        session_tokens=self._session_tokens,
                        uptime_s=round(time.time() - self._session_start, 0))
        logger.info("[MetricsLog] Stopped.")

    def _run(self) -> None:
        """Background loop — sample every interval_s, rotate daily."""
        last_rotate = time.time()
        while not self._stop.wait(timeout=self.interval_s):
            self._sample_and_log()
            if time.time() - last_rotate > 3600:   # check rotation every hour
                self._rotate()
                last_rotate = time.time()


# ── Module-level singleton (set by main.py on startup) ───────────────────────
_metrics_logger: Optional[MetricsLogger] = None


def get_metrics_logger() -> Optional[MetricsLogger]:
    return _metrics_logger


def init_metrics_logger(log_dir: Path, interval_s: float = 30.0, keep_days: int = 7) -> MetricsLogger:
    """Create, start, and register the global MetricsLogger singleton."""
    global _metrics_logger
    _metrics_logger = MetricsLogger(log_dir=log_dir, interval_s=interval_s, keep_days=keep_days)
    _metrics_logger.start()
    return _metrics_logger
