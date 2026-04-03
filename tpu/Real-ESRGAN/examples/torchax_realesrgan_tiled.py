#!/usr/bin/env python3
"""Real-ESRGAN tile-based batch inference on TPU.

Splits the input image into fixed-size tiles, batches them,
runs a single jitted forward pass, then stitches the output.

This approach:
  - Compiles for a fixed tile size (fast, reusable)
  - Batches all tiles in one forward pass (high TPU utilization)
  - Uses halo overlap to eliminate tile boundary artifacts

Usage:
    python torchax_realesrgan_tiled.py [--input-h 2048] [--input-w 1536] [--tile 512] [--halo 16]
"""

import argparse
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))


def patch_conv2d_defaults():
    """Patch torchax conv2d to add default arguments if needed."""
    try:
        import torchax as _tx
        import os
        torchax_path = os.path.dirname(_tx.__file__)
        jaten_file = os.path.join(torchax_path, "ops", "jaten.py")
        with open(jaten_file, "r") as f:
            content = f.read()
        old_sig = "def _aten_conv2d(\n  input,\n  weight,\n  bias,\n  stride,\n  padding,\n  dilation,\n  groups,\n):"
        new_sig = "def _aten_conv2d(\n  input,\n  weight,\n  bias=None,\n  stride=(1, 1),\n  padding=(0, 0),\n  dilation=(1, 1),\n  groups=1,\n):"
        if old_sig in content:
            content = content.replace(old_sig, new_sig)
            with open(jaten_file, "w") as f:
                f.write(content)
            print("  [patch] conv2d defaults added")
        else:
            print("  [patch] conv2d already patched")
    except Exception as e:
        print(f"  [patch] skipped: {e}")


def split_into_tiles(img, tile_size, halo, scale):
    """Split image into overlapping tiles.

    Args:
        img: numpy array (1, C, H, W)
        tile_size: tile size (pixels, input space)
        halo: overlap size (pixels, input space)
        scale: upsampling scale factor

    Returns:
        tiles: list of (tile_array, (y_start, x_start, y_end, x_end)) in input coords
               tile_array is (1, C, tile_h, tile_w)
    """
    _, c, h, w = img.shape
    stride = tile_size - 2 * halo  # effective stride (non-overlapping region)

    tiles = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Tile boundaries in input space
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size  # pull back if at edge
            x_start = x_end - tile_size

            # Clamp to valid range
            y_start = max(0, y_start)
            x_start = max(0, x_start)

            tile = img[:, :, y_start:y_end, x_start:x_end]

            # Pad if tile is smaller than tile_size (edge case for small images)
            th, tw = tile.shape[2], tile.shape[3]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((1, c, tile_size, tile_size), dtype=img.dtype)
                padded[:, :, :th, :tw] = tile
                tile = padded

            tiles.append((tile, (y_start, x_start, y_end, x_end)))

            x += stride
            if x_end >= w:
                break
        y += stride
        if y_end >= h:
            break

    return tiles


def stitch_tiles(tiles_out, coords, output_shape, tile_size, halo, scale):
    """Stitch output tiles back into full image.

    For each tile, only the non-halo center region is used (except at edges).
    This eliminates boundary artifacts.

    Args:
        tiles_out: list of numpy arrays (1, C, tile_h*scale, tile_w*scale)
        coords: list of (y_start, x_start, y_end, x_end) in input space
        output_shape: (1, C, H*scale, W*scale)
        tile_size: tile size in input space
        halo: halo size in input space
        scale: upsampling scale
    """
    result = np.zeros(output_shape, dtype=np.float32)
    _, _, out_h, out_w = output_shape
    _, _, in_h, in_w = output_shape[0], output_shape[1], out_h // scale, out_w // scale

    for tile_out, (y_start, x_start, y_end, x_end) in zip(tiles_out, coords):
        # Determine the crop region within the tile (in output space)
        # Top halo: skip if not at image top
        crop_top = halo * scale if y_start > 0 else 0
        # Bottom halo: skip if not at image bottom
        crop_bot = halo * scale if y_end < in_h else 0
        # Left halo: skip if not at image left
        crop_left = halo * scale if x_start > 0 else 0
        # Right halo: skip if not at image right
        crop_right = halo * scale if x_end < in_w else 0

        tile_h_out = (y_end - y_start) * scale
        tile_w_out = (x_end - x_start) * scale

        # Source region in tile output
        src_y1 = crop_top
        src_y2 = tile_h_out - crop_bot
        src_x1 = crop_left
        src_x2 = tile_w_out - crop_right

        # Destination region in full output
        dst_y1 = y_start * scale + crop_top
        dst_y2 = y_end * scale - crop_bot
        dst_x1 = x_start * scale + crop_left
        dst_x2 = x_end * scale - crop_right

        result[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = \
            tile_out[:, :, src_y1:src_y2, src_x1:src_x2]

    return result


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN tiled TPU inference")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--num-block", type=int, default=12)
    parser.add_argument("--input-h", type=int, default=2048)
    parser.add_argument("--input-w", type=int, default=1536)
    parser.add_argument("--tile", type=int, default=512, help="Tile size in input space")
    parser.add_argument("--halo", type=int, default=16, help="Halo overlap in input space")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--precision", default="highest", choices=["default", "high", "highest"])
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Real-ESRGAN Tiled Inference (TPU)")
    print(f"Input: {args.input_w}x{args.input_h} → Output: {args.input_w*args.scale}x{args.input_h*args.scale}")
    print(f"Tile: {args.tile}x{args.tile}, Halo: {args.halo}px, Precision: {args.precision}")
    print("=" * 60)

    # ─── Step 1: Create model BEFORE torchax ───
    print(f"\n[1/6] Creating RRDBNet (scale={args.scale}, blocks={args.num_block})...")
    from rrdbnet import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=args.scale,
                    num_feat=64, num_block=args.num_block, num_grow_ch=32)
    model.eval()

    if args.weights:
        state_dict = torch.load(args.weights, map_location='cpu')
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=True)
        print(f"  Loaded weights: {args.weights}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,}")

    # ─── Step 2: Initialize torchax ───
    print("[2/6] Initializing torchax...")
    patch_conv2d_defaults()

    import torchax
    from torchax import interop
    import jax

    jax.config.update("jax_default_matmul_precision", args.precision)
    torchax.enable_globally()
    env = torchax.default_env()

    devices = jax.devices()
    print(f"  JAX devices: {len(devices)}x {devices[0].device_kind}")
    print(f"  Precision: {args.precision}")

    # ─── Step 3: Move to TPU ───
    print("[3/6] Moving model to JAX device...")
    model.to("jax")

    # ─── Step 4: Split image into tiles ───
    print(f"[4/6] Splitting input into {args.tile}x{args.tile} tiles (halo={args.halo})...")
    img_np = np.random.rand(1, 3, args.input_h, args.input_w).astype(np.float32)
    tiles_data = split_into_tiles(img_np, args.tile, args.halo, args.scale)
    num_tiles = len(tiles_data)
    print(f"  Tiles: {num_tiles}")

    # Stack into batch
    tile_batch = np.concatenate([t[0] for t in tiles_data], axis=0)  # (N, 3, tile, tile)
    tile_coords = [t[1] for t in tiles_data]
    print(f"  Batch shape: {tile_batch.shape}")

    # ─── Step 5: JIT compile for tile batch ───
    print(f"[5/6] JIT compiling for batch of {num_tiles} tiles ({args.tile}x{args.tile})...")

    def forward_fn(batch_jax):
        with env:
            batch_tx = torchax.tensor.Tensor(batch_jax, env=env)
            with torch.no_grad():
                out = model(batch_tx)
            return interop.jax_view(out)

    jitted_forward = jax.jit(forward_fn)

    batch_jax = jax.numpy.array(tile_batch)

    t0 = time.perf_counter()
    out = jitted_forward(batch_jax)
    jax.block_until_ready(out)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"  Compile + first run: {compile_time:.0f}ms")

    # Warmup
    for _ in range(3):
        out = jitted_forward(batch_jax)
        jax.block_until_ready(out)

    # ─── Step 6: Benchmark ───
    print(f"\n[6/6] Benchmarking {args.runs} iterations (tile batch forward + stitch)...")
    out_shape = (1, 3, args.input_h * args.scale, args.input_w * args.scale)

    fwd_times = []
    total_times = []

    for i in range(args.runs):
        t_total = time.perf_counter()

        # Forward pass (batch)
        t_fwd = time.perf_counter()
        out = jitted_forward(batch_jax)
        jax.block_until_ready(out)
        fwd_ms = (time.perf_counter() - t_fwd) * 1000
        fwd_times.append(fwd_ms)

        # Stitch
        tiles_out_np = np.array(out)
        tiles_out_list = [tiles_out_np[i:i+1] for i in range(num_tiles)]
        result = stitch_tiles(tiles_out_list, tile_coords, out_shape,
                              args.tile, args.halo, args.scale)
        total_ms = (time.perf_counter() - t_total) * 1000
        total_times.append(total_ms)

        if i < 5 or (i + 1) % 5 == 0:
            print(f"  Run {i+1:>2}: fwd={fwd_ms:.2f}ms, total={total_ms:.2f}ms")

    fwd_sorted = sorted(fwd_times)
    total_sorted = sorted(total_times)

    print(f"\n{'═' * 60}")
    print(f"Results: Real-ESRGAN Tiled ({args.input_w}x{args.input_h}, "
          f"tile={args.tile}, halo={args.halo}, {args.precision})")
    print(f"{'═' * 60}")
    print(f"  Tiles: {num_tiles} ({args.tile}x{args.tile})")
    print(f"  Forward pass:")
    print(f"    Average: {sum(fwd_times)/len(fwd_times):.2f} ms")
    print(f"    Median:  {fwd_sorted[len(fwd_times)//2]:.2f} ms")
    print(f"    Min:     {min(fwd_times):.2f} ms")
    print(f"    Max:     {max(fwd_times):.2f} ms")
    print(f"  Total (fwd + stitch):")
    print(f"    Average: {sum(total_times)/len(total_times):.2f} ms")
    print(f"    Median:  {total_sorted[len(total_times)//2]:.2f} ms")
    print(f"    Min:     {min(total_times):.2f} ms")
    print(f"    Max:     {max(total_times):.2f} ms")
    print(f"  Compile:   {compile_time:.0f} ms")
    print(f"  Output:    {result.shape}")
    print(f"{'═' * 60}")

    # Compare with whole-image reference
    print(f"\n  vs whole-image (1536x2048):")
    print(f"    Whole-image highest: ~1650ms")
    print(f"    Whole-image default: ~695ms")
    print(f"    Tile {args.precision}: {total_sorted[len(total_times)//2]:.2f}ms")
    speedup_vs_whole = 1650 / total_sorted[len(total_times)//2] if args.precision == "highest" else 695 / total_sorted[len(total_times)//2]
    print(f"    Speedup: {speedup_vs_whole:.1f}x")


if __name__ == "__main__":
    main()
