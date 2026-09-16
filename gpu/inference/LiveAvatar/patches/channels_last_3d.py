#!/usr/bin/env python3
"""给 LiveAvatar 的 VAE 打 channels_last_3d —— 实测 stream_decode 189.50 → 75.43 ms（2.51×）。

原理：cuDNN 的 3D 卷积内部布局就是 NDHWC。喂 NCDHW 进去，它每一层前后各转一次 ——
profile 里 nchwToNhwc + nhwcToNchw 一次解码调 312 次。转过去之后这些全消失，
而且卷积本身也能直接吃原生布局，所以收益远大于布局转换那 8.6%。

用法（在 LiveAvatar 仓库根目录）：
    python patches/channels_last_3d.py            # 打补丁
    python patches/channels_last_3d.py --revert   # 还原

幂等：重复执行不会重复插入。会先备份成 <file>.orig。
"""
import argparse
import os
import shutil
import sys

TARGET = "liveavatar/models/wan/causal_s2v_pipeline_tpp.py"

ANCHOR_INIT = """        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,dtype=self.param_dtype)"""

PATCH_INIT = ANCHOR_INIT + """
        # ── channels_last_3d：实测 VAE stream_decode 189.5ms → 75.4ms（2.51x）
        # ⚠️ 只能改 5D 权重。对 1D/4D 张量调 channels_last_3d 会直接抛
        #    "required rank 5 tensor"，整个 .to(memory_format=) 挂掉 ——
        #    所以不能简单写 self.vae.model.to(memory_format=...)
        _ncl = 0
        for _mod in self.vae.model.modules():
            for _nm, _p in list(_mod.named_parameters(recurse=False)):
                if _p.dim() == 5:
                    _p.data = _p.data.to(memory_format=torch.channels_last_3d)
                    _ncl += 1
        print(f"[OPT] VAE channels_last_3d 已应用到 {_ncl} 个 5D 权重", flush=True)"""

ANCHOR_INPUT = "                        decode_latents = block_latents.unsqueeze(0)"
PATCH_INPUT = (
    "                        # 输入也要转，否则每次调用还要现转一遍\n"
    "                        decode_latents = block_latents.unsqueeze(0).contiguous(\n"
    "                            memory_format=torch.channels_last_3d)"
)

MARK = "channels_last_3d 已应用到"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    if not os.path.exists(a.file):
        print(f"找不到 {a.file} —— 请在 LiveAvatar 仓库根目录执行", file=sys.stderr)
        return 1

    if a.revert:
        if not os.path.exists(a.file + ".orig"):
            print("没有 .orig 备份，无法还原", file=sys.stderr)
            return 1
        shutil.copy(a.file + ".orig", a.file)
        print("已还原")
        return 0

    src = open(a.file, encoding="utf-8").read()
    if MARK in src:
        print("已经打过了，跳过")
        return 0

    for anchor, name in ((ANCHOR_INIT, "VAE 构造"), (ANCHOR_INPUT, "解码输入")):
        if anchor not in src:
            print(f"锚点不匹配（{name}）—— 上游代码可能变了，不动它", file=sys.stderr)
            return 1

    shutil.copy(a.file, a.file + ".orig")
    src = src.replace(ANCHOR_INIT, PATCH_INIT, 1).replace(ANCHOR_INPUT, PATCH_INPUT, 1)
    open(a.file, "w", encoding="utf-8").write(src)
    print(f"已打补丁（备份在 {a.file}.orig）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
