# tpu_backends.py
"""
TPU Backend implementations for GEMM benchmarking using JAX.

This module is separate from backends.py to avoid mixing PyTorch and JAX dependencies.
TPU backends use JAX/XLA for native TPU support with optimal performance.

Architecture:
    TpuBackendBase (ABC) - Common TPU logic (timing, warmup, JIT compilation)
        ├── TpuV6eBackend - TPU v6e (Trillium) specific implementation
        └── TpuV7Backend  - TPU v7 (Ironwood) specific implementation (future)

Timing Modes:
    - Legacy (use_trace=False): Uses time.perf_counter() for end-to-end timing.
      Includes Python dispatch overhead, typically showing ~65-75% MFU.

    - Trace-based (use_trace=True): Uses JAX profiler to extract pure device
      execution time (device_duration_ps), eliminating Python overhead.
      Typically shows 90%+ MFU for compute-bound workloads.
"""

import abc
import shutil
import statistics
import time
from typing import Tuple, Optional, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .trace_utils import (
    MARKER,
    get_trace,
    get_metrics_from_trace_marker,
    create_temp_trace_dir,
)

# ============================================================================
# JAX dtype mapping (separate from torch dtype system)
# ============================================================================
JAX_DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "int8": jnp.int8,
}

# Output dtype mapping for JAX GEMM operations
# Note: JAX automatically handles accumulator precision, but we track output type for bandwidth calculation
JAX_DTYPE_OUTPUT_MAP = {
    jnp.float32: jnp.float32,
    jnp.float16: jnp.float16,
    jnp.bfloat16: jnp.bfloat16,
    jnp.int8: jnp.int32,  # int8 matmul accumulates to int32
}


def parse_jax_dtype(dtype_str: str) -> jnp.dtype:
    """Convert string to JAX dtype."""
    if dtype_str not in JAX_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: '{dtype_str}'. Supported: {list(JAX_DTYPE_MAP.keys())}")
    return JAX_DTYPE_MAP[dtype_str]


def get_jax_output_dtype(input_dtype: jnp.dtype) -> jnp.dtype:
    """Get output dtype for a given input dtype."""
    # Handle both dtype objects and their string representations
    for key, value in JAX_DTYPE_OUTPUT_MAP.items():
        if jnp.dtype(key) == jnp.dtype(input_dtype):
            return value
    return input_dtype  # fallback to same dtype


# ============================================================================
# GEMM kernel implementations (JIT-compiled for TPU)
# ============================================================================

@partial(jax.jit, static_argnums=(3,))
def _gemm_kernel(a: jnp.ndarray, b: jnp.ndarray, key: jax.random.PRNGKey,
                 output_dtype: jnp.dtype) -> jnp.ndarray:
    """
    JIT-compiled GEMM kernel using jax.lax.dot_general.

    Uses dot_general for precise control over contraction dimensions.
    This is the low-level operation that maps directly to TPU MXU.

    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N)
        key: Unused, but needed for consistent function signature with random ops
        output_dtype: Output data type (for int8->int32 accumulation)

    Returns:
        Result matrix (M, N)
    """
    # Standard matrix multiplication: contract K dimension
    # dimension_numbers: ((lhs_contracting), (rhs_contracting)), ((lhs_batch), (rhs_batch))
    dimension_numbers = (((1,), (0,)), ((), ()))

    result = lax.dot_general(
        a, b,
        dimension_numbers=dimension_numbers,
        preferred_element_type=output_dtype,  # Control accumulator/output precision
    )
    return result


@partial(jax.jit, static_argnums=(3,))
def _gemm_kernel_with_marker(a: jnp.ndarray, b: jnp.ndarray, key: jax.random.PRNGKey,
                              output_dtype: jnp.dtype) -> jnp.ndarray:
    """
    JIT-compiled GEMM kernel with MARKER for trace-based timing.

    The jax.named_scope(MARKER) MUST be inside the jit function for the marker
    to appear in the tf_op field of trace events. This is critical for accurate
    trace-based timing extraction.

    Reference: accelerator-microbenchmarks/Ironwood/src/benchmark_gemm.py
    """
    # MARKER must be inside the jit function to appear in tf_op field
    with jax.named_scope(MARKER):
        dimension_numbers = (((1,), (0,)), ((), ()))
        result = lax.dot_general(
            a, b,
            dimension_numbers=dimension_numbers,
            preferred_element_type=output_dtype,
        )
    return result


def _create_input_tensors_jax(m: int, n: int, k: int, dtype: jnp.dtype,
                               key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create input tensors on TPU with appropriate initialization."""
    key1, key2 = jax.random.split(key)

    if jnp.issubdtype(dtype, jnp.floating):
        # Floating point: random normal values
        a = jax.random.normal(key1, (m, k), dtype=dtype)
        b = jax.random.normal(key2, (k, n), dtype=dtype)
    elif jnp.issubdtype(dtype, jnp.integer):
        # Integer: random values in appropriate range
        a = jax.random.randint(key1, (m, k), -128, 127, dtype=dtype)
        b = jax.random.randint(key2, (k, n), -128, 127, dtype=dtype)
    else:
        # Fallback: ones
        a = jnp.ones((m, k), dtype=dtype)
        b = jnp.ones((k, n), dtype=dtype)

    return a, b


# ============================================================================
# TPU Backend Base Class
# ============================================================================

class TpuBackendBase(abc.ABC):
    """
    Abstract base class for TPU backends.

    Provides common functionality:
    - JAX device detection and selection
    - Warmup and profiling loop structure
    - Timing using block_until_ready() for accurate measurement
    - JIT compilation management
    - Trace-based timing for accurate MFU measurement

    Subclasses implement:
    - get_device_name(): Return human-readable device name
    - get_tpu_generation(): Return TPU generation string (e.g., "v6e", "v7")
    - get_theoretical_peak(): Return theoretical TFLOPS for a given dtype
    """

    def __init__(self, warmup_iter: int = 10, prof_iter: int = 100, use_trace: bool = True):
        """
        Initialize TPU backend.

        Args:
            warmup_iter: Number of warmup iterations (JIT compilation + cache warming)
            prof_iter: Number of profiling iterations for timing measurement
            use_trace: If True, use JAX profiler trace for accurate timing (90%+ MFU).
                      If False, use time.perf_counter() which includes Python overhead.
        """
        self.warmup_iter = warmup_iter
        self.prof_iter = prof_iter
        self.use_trace = use_trace

        # Verify TPU is available
        self._verify_tpu_available()

        # Get device info
        self.device = jax.devices('tpu')[0]
        self.device_name = self.get_device_name()
        self.device_str = "tpu"

        # Random key for reproducible tensor generation
        self._rng_key = jax.random.PRNGKey(42)

    def _verify_tpu_available(self):
        """Check if TPU is available."""
        try:
            devices = jax.devices('tpu')
            if not devices:
                raise RuntimeError("No TPU devices found")
        except RuntimeError as e:
            raise RuntimeError(f"TPU not available: {e}")

    @abc.abstractmethod
    def get_device_name(self) -> str:
        """Return human-readable device name (e.g., 'Google TPU v6e')."""
        pass

    @abc.abstractmethod
    def get_tpu_generation(self) -> str:
        """Return TPU generation identifier (e.g., 'v6e', 'v7')."""
        pass

    def get_device_str(self) -> str:
        """Return device string for config lookup."""
        return "tpu"

    def synchronize(self):
        """
        Synchronize TPU execution.

        In JAX, we use block_until_ready() on arrays rather than a global sync.
        This method is kept for API compatibility with GPU backends.
        """
        # JAX handles sync via block_until_ready() on results
        # This is a no-op for interface compatibility
        pass

    def run(self, m: int, n: int, k: int, dtype) -> float:
        """
        Run GEMM benchmark for given dimensions and dtype.

        Args:
            m: Number of rows in A and C
            n: Number of columns in B and C
            k: Shared dimension (columns of A, rows of B)
            dtype: Can be JAX dtype, torch dtype, or string

        Returns:
            Average execution time in microseconds, or -1.0 on error
        """
        if self.use_trace:
            return self._run_with_trace(m, n, k, dtype)
        else:
            return self._run_legacy(m, n, k, dtype)

    def _run_legacy(self, m: int, n: int, k: int, dtype) -> float:
        """
        Run GEMM benchmark using legacy time.perf_counter() timing.

        This method includes Python dispatch overhead in timing measurements,
        typically resulting in ~65-75% MFU for compute-bound workloads.
        """
        try:
            # Handle dtype conversion (support torch.dtype, jax dtype, or string)
            jax_dtype = self._convert_dtype(dtype)
            output_dtype = get_jax_output_dtype(jax_dtype)

            # Get new random key for this run
            self._rng_key, subkey = jax.random.split(self._rng_key)

            # Create input tensors on TPU
            a, b = _create_input_tensors_jax(m, n, k, jax_dtype, subkey)

            # Ensure tensors are on TPU and materialized
            a = jax.device_put(a, self.device)
            b = jax.device_put(b, self.device)
            a.block_until_ready()
            b.block_until_ready()

            # Warmup phase (includes JIT compilation on first call)
            for _ in range(self.warmup_iter):
                result = _gemm_kernel(a, b, subkey, output_dtype)
                result.block_until_ready()

            # Profiling phase with accurate timing
            start_time = time.perf_counter()
            for _ in range(self.prof_iter):
                result = _gemm_kernel(a, b, subkey, output_dtype)
                result.block_until_ready()  # Critical: wait for TPU to finish
            end_time = time.perf_counter()

            # Calculate average time in microseconds
            total_time_s = end_time - start_time
            avg_time_us = (total_time_s * 1_000_000) / self.prof_iter

            return avg_time_us

        except Exception as e:
            print(f"  > [TPU Error] m={m}, n={n}, k={k}, dtype={dtype}: {e}")
            return -1.0

    def _run_with_trace(self, m: int, n: int, k: int, dtype) -> float:
        """
        Run GEMM benchmark using JAX profiler trace-based timing.

        This method extracts pure device execution time (device_duration_ps)
        from JAX profiler traces, eliminating Python overhead. This typically
        results in 90%+ MFU for compute-bound workloads.

        The key technique is using jax.named_scope(MARKER) to tag operations,
        then parsing the trace to find events with that marker.
        """
        trace_dir = None
        try:
            # Handle dtype conversion
            jax_dtype = self._convert_dtype(dtype)
            output_dtype = get_jax_output_dtype(jax_dtype)

            # Get new random key for this run
            self._rng_key, subkey = jax.random.split(self._rng_key)

            # Create input tensors on TPU
            a, b = _create_input_tensors_jax(m, n, k, jax_dtype, subkey)

            # Ensure tensors are on TPU and materialized
            a = jax.device_put(a, self.device)
            b = jax.device_put(b, self.device)
            a.block_until_ready()
            b.block_until_ready()

            # Warmup phase (includes JIT compilation on first call)
            # Use the MARKER kernel so JIT compiles the traced version
            for _ in range(self.warmup_iter):
                result = _gemm_kernel_with_marker(a, b, subkey, output_dtype)
                result.block_until_ready()

            # Create temporary trace directory
            trace_dir = create_temp_trace_dir()

            # Profiling phase with trace collection
            # CRITICAL: The MARKER must be INSIDE the jitted function (not outside)
            # for it to appear in the tf_op field of trace events.
            # Reference: accelerator-microbenchmarks/Ironwood/src/benchmark_gemm.py
            with jax.profiler.trace(trace_dir):
                for i in range(self.prof_iter):
                    # Use jax.profiler.StepTraceAnnotation for proper step tracking
                    with jax.profiler.StepTraceAnnotation("gemm", step_num=i):
                        result = _gemm_kernel_with_marker(a, b, subkey, output_dtype)
                        result.block_until_ready()

            # Extract timing from trace
            trace = get_trace(trace_dir)
            durations_ms = get_metrics_from_trace_marker(trace, MARKER)

            if not durations_ms:
                print(f"  > [Warning] No trace events found, falling back to legacy timing")
                return self._run_legacy(m, n, k, dtype)

            # Calculate median time in microseconds (more robust than mean)
            median_time_ms = statistics.median(durations_ms)
            avg_time_us = median_time_ms * 1000  # Convert ms to us

            return avg_time_us

        except Exception as e:
            print(f"  > [TPU Trace Error] m={m}, n={n}, k={k}, dtype={dtype}: {e}")
            # Fallback to legacy timing on error
            print(f"  > Falling back to legacy timing...")
            try:
                return self._run_legacy(m, n, k, dtype)
            except Exception:
                return -1.0

        finally:
            # Always clean up trace directory to prevent resource leak
            if trace_dir is not None:
                shutil.rmtree(trace_dir, ignore_errors=True)

    def _convert_dtype(self, dtype) -> jnp.dtype:
        """
        Convert various dtype representations to JAX dtype.

        Handles:
        - String: "float32", "bfloat16", etc.
        - JAX dtype: jnp.float32, etc.
        - PyTorch dtype: torch.float32, etc. (for compatibility with main.py)
        """
        if isinstance(dtype, str):
            return parse_jax_dtype(dtype)

        # Check if it's already a JAX dtype
        try:
            return jnp.dtype(dtype)
        except TypeError:
            pass

        # Try to handle torch dtype by name matching
        dtype_name = str(dtype).replace("torch.", "")
        if dtype_name in JAX_DTYPE_MAP:
            return JAX_DTYPE_MAP[dtype_name]

        raise ValueError(f"Cannot convert dtype: {dtype}")


# ============================================================================
# TPU v6e Backend Implementation
# ============================================================================

class TpuV6eBackend(TpuBackendBase):
    """
    TPU v6e (Trillium) backend implementation.

    Hardware specs:
    - Peak BF16: 918 TFLOPS
    - HBM: 32 GB @ 1,600 GB/s
    - MXU: 256x256, 2 per TensorCore
    - TensorCore: 1 per chip

    Optimal matrix dimensions:
    - Prefer dimensions divisible by 128 (MXU tiling)
    - For maximum MXU utilization, use multiples of 256

    Note on float32 performance:
    - TPU MXU natively executes in bfloat16
    - float32 GEMM uses bf16 compute with fp32 accumulation
    - Therefore float32 achieves same compute throughput as bf16
    """

    # Theoretical peak performance (TFLOPS)
    # Note: float32 uses bf16 compute path on TPU, so same peak as bf16
    PEAK_TFLOPS = {
        "bfloat16": 918.0,
        "float16": 918.0,   # Same as bf16 on TPU
        "float32": 918.0,   # Uses bf16 compute with fp32 accumulation
        "int8": 1836.0,     # 2x bf16 for INT8
    }

    # Memory bandwidth (GB/s)
    HBM_BANDWIDTH = 1600.0

    def get_device_name(self) -> str:
        """Return device name."""
        return "Google TPU v6e (Trillium)"

    def get_tpu_generation(self) -> str:
        """Return TPU generation."""
        return "v6e"

    def get_theoretical_peak(self, dtype_str: str) -> float:
        """Get theoretical peak TFLOPS for a given dtype."""
        return self.PEAK_TFLOPS.get(dtype_str, 918.0)


# ============================================================================
# TPU v7 Backend Implementation (Ironwood - Dual-Chiplet Architecture)
# ============================================================================

class TpuV7Backend(TpuBackendBase):
    """
    TPU v7 (Ironwood) backend implementation.

    Hardware specs (per chip, from Google Cloud docs 2026-02-09):
    - Peak BF16: 2,307 TFLOPS
    - Peak FP8:  4,614 TFLOPS (native FP8 support)
    - HBM: 192 GiB @ 7,380 GB/s
    - MXU: 2 TensorCores per chip, 4 SparseCores per chip
    - ICI: 1,200 GBps bidirectional

    Dual-Chiplet Architecture:
    - Each physical chip contains 2 chiplets
    - JAX exposes each chip as 2 separate devices (one per chiplet)
    - Each chiplet: 1 TensorCore, 2 SparseCores, 96 GB HBM
    - Per-chiplet peak: BF16 1,153.5 TFLOPS, FP8 2,307 TFLOPS, HBM BW 3,690 GB/s
    - Chiplets connected via die-to-die (D2D) interface (6x faster than 1D ICI)

    Note on float32 performance:
    - TPU MXU natively executes in bfloat16
    - float32 GEMM uses bf16 compute with fp32 accumulation
    - Therefore float32 achieves same compute throughput as bf16
    """

    # Per-chip peak performance (TFLOPS)
    # Source: https://docs.cloud.google.com/tpu/docs/tpu7x
    CHIP_PEAK_TFLOPS = {
        "bfloat16": 2307.0,
        "float16": 2307.0,   # Same MXU throughput as bf16
        "float32": 2307.0,   # Uses bf16 compute with fp32 accumulation
        "int8": 4614.0,      # 2x bf16 (same as FP8)
    }

    # Per-chiplet (JAX device) peak performance (TFLOPS)
    # JAX exposes each chiplet as a separate device on v7
    PEAK_TFLOPS = {
        "bfloat16": 1153.5,  # 2307 / 2 chiplets
        "float16": 1153.5,
        "float32": 1153.5,
        "int8": 2307.0,      # 4614 / 2 chiplets
    }

    # Memory bandwidth (GB/s) - per chip: 7380, per chiplet: 3690
    HBM_BANDWIDTH = 3690.0       # Per chiplet (JAX device)
    HBM_BANDWIDTH_CHIP = 7380.0  # Per physical chip
    HBM_CAPACITY_GIB = 192       # Per chip (96 GiB per chiplet)

    def get_device_name(self) -> str:
        """Return device name."""
        return "Google TPU v7 (Ironwood)"

    def get_tpu_generation(self) -> str:
        """Return TPU generation."""
        return "v7"

    def get_theoretical_peak(self, dtype_str: str) -> float:
        """
        Get theoretical peak TFLOPS for a given dtype.

        Returns per-chiplet (per JAX device) peak since the benchmark
        runs on a single JAX device, which is one chiplet on v7.
        """
        return self.PEAK_TFLOPS.get(dtype_str, 1153.5)


# ============================================================================
# Factory function for TPU backend selection
# ============================================================================

def detect_tpu_backend(
    warmup_iter: int = 10,
    prof_iter: int = 100,
    use_trace: bool = True
) -> Optional[TpuBackendBase]:
    """
    Auto-detect TPU generation and return appropriate backend.

    Detection strategy:
    1. Check if TPU is available via jax.devices('tpu')
    2. Parse device platform version to determine generation
    3. Check number of devices vs chips (dual-chiplet detection for v7)
    4. Return matching backend instance

    Args:
        warmup_iter: Number of warmup iterations
        prof_iter: Number of profiling iterations
        use_trace: If True, use trace-based timing for accurate MFU (default: True)

    Returns:
        TpuBackendBase subclass instance, or None if no TPU available
    """
    try:
        devices = jax.devices('tpu')
        if not devices:
            return None

        device = devices[0]

        # Collect all detection signals
        device_str = str(device).lower()
        platform_version = getattr(device, 'platform_version', '').lower()
        device_kind = getattr(device, 'device_kind', '').lower()

        # Log detection info for debugging
        print(f"[TPU Detection] device_str: {device_str}")
        print(f"[TPU Detection] platform_version: {platform_version}")
        print(f"[TPU Detection] device_kind: {device_kind}")
        print(f"[TPU Detection] num_devices: {len(devices)}")

        # v7 (Ironwood) detection - multiple strategies:
        # 1. Direct string match in device info
        # 2. Check for 'tpu7' in platform version or device kind
        # 3. Dual-chiplet detection: v7 has 2 devices per chip (core_on_chip dimension)
        is_v7 = False

        # Strategy 1: String matching
        all_info = f"{device_str} {platform_version} {device_kind}"
        if any(marker in all_info for marker in ['v7', 'ironwood', 'tpu7']):
            is_v7 = True

        # Strategy 2: Check coords dimensionality (v7 uses 4D coords for dual-chiplet)
        if not is_v7:
            try:
                # On v7, JAX device coords have 4 dimensions (x, y, z, chiplet)
                coords = getattr(device, 'coords', None)
                if coords is not None and len(coords) == 4:
                    is_v7 = True
                    print(f"[TPU Detection] 4D coords detected (dual-chiplet): {coords}")
            except Exception:
                pass

        if is_v7:
            print(f"[TPU Detection] Detected TPU v7 (Ironwood)")
            print(f"[TPU Detection] Dual-chiplet: each JAX device = 1 chiplet (half a physical chip)")
            return TpuV7Backend(warmup_iter, prof_iter, use_trace)

        # v6e detection (Trillium)
        if any(marker in all_info for marker in ['v6', 'trillium']):
            print(f"[TPU Detection] Detected TPU v6e (Trillium)")
            return TpuV6eBackend(warmup_iter, prof_iter, use_trace)

        # Default to v6e for unknown TPU versions
        print(f"[Info] Unknown TPU version detected ({device_str}), defaulting to v6e backend")
        return TpuV6eBackend(warmup_iter, prof_iter, use_trace)

    except Exception as e:
        print(f"[Info] TPU detection failed: {e}")
        return None


# ============================================================================
# Utility: Check if TPU is available (for use in main.py)
# ============================================================================

def is_tpu_available() -> bool:
    """Check if TPU is available."""
    try:
        devices = jax.devices('tpu')
        return len(devices) > 0
    except (RuntimeError, AttributeError, Exception):
        # RuntimeError: JAX backend not available
        # AttributeError: devices function not available
        return False
