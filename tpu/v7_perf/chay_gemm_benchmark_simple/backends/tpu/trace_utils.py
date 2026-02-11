# trace_utils.py
"""
Trace-based timing utilities for TPU GEMM benchmarking.

This module provides functions to extract accurate device execution times
from JAX profiler traces, eliminating Python overhead from measurements.

The key insight is that device_duration_ps in the trace represents pure
TPU execution time, giving much more accurate MFU calculations (90%+ vs 65%).

Reference: accelerator-microbenchmarks/Ironwood/src/benchmark_utils.py
"""

import gzip
import json
import os
import pathlib
import random
import string
import tempfile
from typing import Any, Dict, List, Optional

# MARKER string used to identify our GEMM operations in the trace
MARKER = "!!MARKER!!"


def get_trace(log_dir: str) -> Dict[str, Any]:
    """
    Extract the trace object from the JAX profiler log directory.

    The trace is stored as a gzipped JSON file under:
    {log_dir}/plugins/profile/{timestamp}/*.trace.json.gz

    Args:
        log_dir: Path to the JAX profiler output directory

    Returns:
        Trace dictionary containing traceEvents

    Raises:
        ValueError: If trace folder structure is invalid
        FileNotFoundError: If trace files are missing
    """
    # Navigate to the folder with the latest trace dump
    profile_path = pathlib.Path(log_dir).absolute() / "plugins" / "profile"

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile directory not found: {profile_path}")

    trace_folders = list(profile_path.iterdir())
    if not trace_folders:
        raise FileNotFoundError(f"No trace folders found in: {profile_path}")

    # Get the most recently modified trace folder
    latest_trace_folder = max(trace_folders, key=os.path.getmtime)

    # Find the trace.json.gz file
    trace_jsons = list(latest_trace_folder.glob("*.trace.json.gz"))

    if len(trace_jsons) != 1:
        raise ValueError(
            f"Expected exactly 1 trace.json.gz file in {latest_trace_folder}, "
            f"found {len(trace_jsons)}"
        )

    trace_json = trace_jsons[0]

    with gzip.open(trace_json, "rb") as f:
        trace = json.load(f)

    return trace


def get_metrics_from_trace_marker(
    trace: Dict[str, Any],
    marker: str = MARKER
) -> List[float]:
    """
    Extract device execution times from trace events marked with MARKER.

    This function finds all trace events that contain the MARKER in their
    tf_op field and extracts the device_duration_ps (picoseconds) value.

    On TPU v7 (Ironwood), jax.named_scope() writes the marker to tf_op field.
    On TPU v6e (Trillium), the marker may not appear, so we fall back to
    finding dot_general operations directly.

    Args:
        trace: Trace dictionary from get_trace()
        marker: Marker string to search for (default: MARKER)

    Returns:
        List of execution times in milliseconds

    Raises:
        KeyError: If trace doesn't contain expected fields
    """
    if "traceEvents" not in trace:
        raise KeyError("Key 'traceEvents' not found in trace.")

    # Strategy 1: Find events containing MARKER in tf_op (works on v7)
    marker_events = []
    for event in trace["traceEvents"]:
        args = event.get("args", {})
        tf_op = args.get("tf_op", "")
        if marker in tf_op:
            marker_events.append(event)

    # Strategy 2: If no MARKER events, fall back to finding dot_general ops (for v6e)
    if not marker_events:
        print(f"[Trace] No MARKER events in tf_op, trying fallback strategy...")
        marker_events = _find_gemm_events_fallback(trace)

    if not marker_events:
        print(f"[Warning] No GEMM events found in trace")
        return []

    # Filter for "call-done" events if available (for sparse core offloading)
    call_done_events = [
        e for e in marker_events if e.get("name", "").endswith("call-done")
    ]
    if call_done_events:
        marker_events = call_done_events

    # Get unique PIDs (each TPU device has a different PID)
    unique_pids = set(e["pid"] for e in marker_events)

    # Use events from the device with the smallest PID (TPU-0)
    min_pid = min(unique_pids)
    events_from_min_pid = [e for e in marker_events if e["pid"] == min_pid]

    # Extract device_duration_ps and convert to milliseconds
    durations_ms = []
    for e in events_from_min_pid:
        device_duration_ps = e.get("args", {}).get("device_duration_ps")
        if device_duration_ps is not None:
            # Convert picoseconds to milliseconds: ps / 1e9 = ms
            durations_ms.append(float(device_duration_ps) / 1e9)
        elif "dur" in e:
            # Fallback: use dur field (microseconds) and convert to ms
            durations_ms.append(float(e["dur"]) / 1e3)

    print(f"[Trace] Collected {len(durations_ms)} timing samples from device (pid={min_pid})")

    return durations_ms


def _find_gemm_events_fallback(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fallback strategy to find GEMM-related events when MARKER is not present.

    On TPU v6e, jax.named_scope() doesn't write to tf_op field. Instead, we
    look for fusion events that contain dot_general operations.

    Args:
        trace: Trace dictionary

    Returns:
        List of events that appear to be GEMM operations
    """
    gemm_events = []

    for event in trace["traceEvents"]:
        args = event.get("args", {})
        name = event.get("name", "")
        tf_op = args.get("tf_op", "")

        # Check if this is a dot_general (GEMM) operation
        # On v6e, these appear as "fusion" events with tf_op containing "dot_general"
        if "dot_general" in tf_op:
            if "device_duration_ps" in args:
                gemm_events.append(event)

    if gemm_events:
        print(f"[Trace] Found {len(gemm_events)} dot_general events via fallback")

    return gemm_events


def generate_trace_name(prefix: str = "gemm") -> str:
    """
    Generate a unique trace directory name.

    Args:
        prefix: Prefix for the trace name

    Returns:
        Unique trace name string
    """
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"{prefix}_{suffix}"


def create_temp_trace_dir() -> str:
    """
    Create a temporary directory for trace output.

    Returns:
        Path to the temporary directory
    """
    trace_dir = tempfile.mkdtemp(prefix="tpu_gemm_trace_")
    return trace_dir
