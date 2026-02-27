#!/usr/bin/env python3
"""Common torchax compatibility fixes for YOLO models.

This file documents all compatibility issues encountered when running
YOLO on TPU via torchax, with self-contained fix functions.

Each fix is independent and can be applied selectively.
"""

import types
import torch


# ═══════════════════════════════════════════════════════════════
# Fix 1: torch.arange keyword argument compatibility
# ═══════════════════════════════════════════════════════════════

def fix_torch_arange():
    """Fix torchax handling of torch.arange(end=N) keyword-only calls.

    Problem:
        Some code calls torch.arange(end=10) instead of torch.arange(10).
        Certain torchax versions don't handle the keyword-only form correctly.

    Solution:
        Monkey-patch torch.arange to normalize the call signature.
    """
    _orig_arange = torch.arange

    def _patched_arange(*args, **kwargs):
        if "end" in kwargs and "start" not in kwargs and len(args) == 0:
            end = kwargs.pop("end")
            return _orig_arange(0, end, **kwargs)
        return _orig_arange(*args, **kwargs)

    torch.arange = _patched_arange
    print("[Fix 1] torch.arange patched for keyword arg compatibility")


# ═══════════════════════════════════════════════════════════════
# Fix 2: conv2d default arguments in torchax
# ═══════════════════════════════════════════════════════════════

def fix_conv2d_defaults(torchax_path=None):
    """Fix missing default values in torchax's _aten_conv2d implementation.

    Problem:
        torchax v0.0.11's _aten_conv2d requires all arguments positionally,
        but PyTorch's ATen schema defines defaults for stride, padding,
        dilation, and groups. When the dispatcher doesn't pass these,
        we get: TypeError: _aten_conv2d() missing 2 required positional
        arguments: 'dilation' and 'groups'

    Solution:
        Patch the torchax source to add default values.

    Args:
        torchax_path: Path to torchax installation. Auto-detected if None.
    """
    if torchax_path is None:
        import torchax as _tx
        import os
        torchax_path = os.path.dirname(_tx.__file__)

    jaten_file = os.path.join(torchax_path, "ops", "jaten.py")

    try:
        with open(jaten_file, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[Fix 2] SKIP: {jaten_file} not found")
        return

    old_sig = """def _aten_conv2d(
  input,
  weight,
  bias,
  stride,
  padding,
  dilation,
  groups,
):"""

    new_sig = """def _aten_conv2d(
  input,
  weight,
  bias=None,
  stride=(1, 1),
  padding=(0, 0),
  dilation=(1, 1),
  groups=1,
):"""

    if old_sig in content:
        content = content.replace(old_sig, new_sig)
        with open(jaten_file, "w") as f:
            f.write(content)
        print(f"[Fix 2] conv2d defaults patched in {jaten_file}")
    else:
        print("[Fix 2] SKIP: conv2d signature already patched or different format")


# ═══════════════════════════════════════════════════════════════
# Fix 3: Fuse BatchNorm to eliminate buffer issues
# ═══════════════════════════════════════════════════════════════

def fix_batchnorm_buffers(model):
    """Fuse BatchNorm layers into Conv layers.

    Problem:
        YOLO has many BatchNorm layers with registered buffers
        (running_mean, running_var, num_batches_tracked).
        After model.to("jax"), these buffers may remain as plain
        torch.Tensor, causing AssertionError in torchax.compile's
        JittableModule when it calls jax_view(self.buffers).

    Solution:
        Call model.fuse() which folds BN into preceding Conv layers,
        completely eliminating BN buffers.

    Args:
        model: YOLO model object (from ultralytics.YOLO)

    Returns:
        The fused model (same object, modified in-place)
    """
    model.model.eval()

    # Count buffers before
    n_before = sum(1 for _ in model.model.named_buffers())

    model.model.fuse()

    n_after = sum(1 for _ in model.model.named_buffers())
    print(f"[Fix 3] BN fused: {n_before} buffers → {n_after} buffers")

    return model


# ═══════════════════════════════════════════════════════════════
# Fix 4: Precompute anchors and patch dynamic tensor creation
# ═══════════════════════════════════════════════════════════════

def fix_dynamic_anchors(model, sample_input):
    """Precompute YOLO anchors and remove dynamic creation from forward.

    Problem:
        YOLO's Detect head calls make_anchors() inside _get_decode_boxes(),
        which creates tensors dynamically using torch.arange/meshgrid.
        Inside jax.jit, this causes:
        - ConcretizationTypeError (shape comparison on traced values)
        - Side effects (self.shape = shape mutation)
        - Re-tracing on every call

    Solution:
        1. Run one forward pass to trigger anchor computation
        2. Cache the anchors as model attributes
        3. Monkey-patch _get_decode_boxes to skip the dynamic check

    Args:
        model: YOLO inner model (model.model)
        sample_input: A sample input tensor on JAX device

    Note:
        This fix requires FIXED input size. If input dimensions change,
        call this function again with the new size.
    """
    import jax

    detect = model.model[-1]  # Last module is Detect head

    # Trigger anchor computation
    with torch.no_grad():
        _ = model(sample_input)
        jax.effects_barrier()

    # Cache current anchors
    cached_shape = detect.shape

    # Patch: always use cached anchors, skip shape check
    def patched_get_decode_boxes(self, x):
        dbox = self.decode_bboxes(
            self.dfl(x["boxes"]),
            self.anchors.unsqueeze(0)
        ) * self.strides
        return dbox

    detect._get_decode_boxes = types.MethodType(patched_get_decode_boxes, detect)
    print(f"[Fix 4] Anchors precomputed for input shape {cached_shape}")
    print(f"        Anchors: {detect.anchors.shape}, Strides: {detect.strides.shape}")


# ═══════════════════════════════════════════════════════════════
# Fix 5: Move remaining buffers to JAX device
# ═══════════════════════════════════════════════════════════════

def fix_buffer_devices(model):
    """Ensure all registered buffers are on the JAX device.

    Problem:
        model.to("jax") moves parameters but some buffers may remain
        on CPU, especially after model modifications or when modules
        have non-standard buffer registration.

    Solution:
        Iterate all named_buffers and move each to "jax" device.

    Args:
        model: YOLO inner model (model.model)
    """
    moved = 0
    for name, buf in model.named_buffers():
        if str(buf.device) != "jax":
            parts = name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            setattr(obj, parts[-1], buf.to("jax"))
            moved += 1

    print(f"[Fix 5] Moved {moved} buffers to JAX device")


# ═══════════════════════════════════════════════════════════════
# Convenience: Apply all fixes at once
# ═══════════════════════════════════════════════════════════════

def apply_all_fixes(yolo_model, sample_input):
    """Apply all compatibility fixes for YOLO + torchax.

    Usage:
        from torchax_common_fixes import apply_all_fixes

        model = YOLO("yolo11n.pt")
        torchax.enable_globally()
        model.to("jax")
        img_jax = preprocess(image).to("jax")

        apply_all_fixes(model, img_jax)
        # Model is now ready for jax.jit compilation

    Args:
        yolo_model: YOLO model object (from ultralytics.YOLO)
        sample_input: A preprocessed input tensor on JAX device
    """
    print("Applying all torchax compatibility fixes for YOLO...")
    print()

    fix_torch_arange()
    fix_batchnorm_buffers(yolo_model)
    yolo_model.model.to("jax")
    fix_buffer_devices(yolo_model.model)
    fix_dynamic_anchors(yolo_model.model, sample_input)

    print()
    print("All fixes applied. Model is ready for jax.jit compilation.")


# ═══════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("torchax_common_fixes.py - YOLO compatibility fix collection")
    print()
    print("Available fixes:")
    print("  fix_torch_arange()          - torch.arange keyword arg compat")
    print("  fix_conv2d_defaults()       - conv2d missing default args")
    print("  fix_batchnorm_buffers()     - fuse BN, eliminate buffer issues")
    print("  fix_dynamic_anchors()       - precompute anchors, patch Detect head")
    print("  fix_buffer_devices()        - move all buffers to JAX device")
    print("  apply_all_fixes()           - apply all of the above")
    print()
    print("Usage:")
    print("  from torchax_common_fixes import apply_all_fixes")
    print("  apply_all_fixes(yolo_model, sample_input)")
