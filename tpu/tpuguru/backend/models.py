"""模型注册表 + MoE 后端注册表。

**为什么要先选模型**：后面每一件事都依赖形状 ——
tile 上界看 `hidden`/`mlp`，整除类 lint 看 `num_experts`，
显存估算看参数量，能不能开专家维分片看专家数能不能被 FSDP 整除。
形状不知道，这些全部退化成猜。

**provenance 字段是认真的**：
  `measured`  我们自己在 v7 上跑过，形状与实测数字都可信
  `public`    公开 config，形状可信、但 TPU 上的实测数字我们没有
不要把这两类混在一起报。
"""

from __future__ import annotations

# ── 模型 ────────────────────────────────────────────────────────
# tile_default 只对 MoE 分组矩阵乘有意义：(batch_seq, embed, mlp)
# 经验起点：embed 取 min(2048, hidden/2)（取满会撞 Mosaic 向量化限制），mlp 取满
MODELS: dict[str, dict] = {
    # —— 我们自己加进 MaxText 的 ——
    "hunyuan3-295b": {
        "family": "hunyuan3", "label": "Hunyuan3-295B-A21B", "moe": True,
        "layers": 80, "num_experts": 192, "top_k": 8, "hidden": 4096, "mlp": 1536,
        "vocab": 120832, "params_b": 298.786, "act_params_b": 21.0,
        "tile_default": [512, 2048, 1536], "provenance": "measured",
        "note": "我们后加进 MaxText 的，主线没有。64 芯片 v7 上完整调过一轮。",
    },
    # —— MaxText 主线自带 ——
    "deepseek3-671b": {
        "family": "deepseek", "label": "DeepSeek-V3 671B", "moe": True,
        "layers": 61, "num_experts": 256, "top_k": 8, "hidden": 7168, "mlp": 2048,
        "vocab": 129280, "params_b": 671.0, "act_params_b": 37.0,
        "tile_default": [512, 2048, 2048], "provenance": "public",
        "note": "MLA 注意力 + 细粒度专家。v7 上 attention 用 dot_product，"
                "flash+splash 实测编译 70 分钟不出来。",
    },
    "deepseek2-16b": {
        "family": "deepseek", "label": "DeepSeek-V2-Lite 16B", "moe": True,
        "layers": 27, "num_experts": 64, "top_k": 6, "hidden": 2048, "mlp": 1408,
        "vocab": 102400, "params_b": 15.7, "act_params_b": 2.4,
        "tile_default": [512, 1024, 1408], "provenance": "public",
        "note": "小规模冒烟用，几张卡就能跑。",
    },
    "qwen3-235b-a22b": {
        "family": "qwen3", "label": "Qwen3-235B-A22B", "moe": True,
        "layers": 94, "num_experts": 128, "top_k": 8, "hidden": 4096, "mlp": 1536,
        "vocab": 151936, "params_b": 235.0, "act_params_b": 22.0,
        "tile_default": [512, 2048, 1536], "provenance": "public",
    },
    "qwen3-30b-a3b": {
        "family": "qwen3", "label": "Qwen3-30B-A3B", "moe": True,
        "layers": 48, "num_experts": 128, "top_k": 8, "hidden": 2048, "mlp": 768,
        "vocab": 151936, "params_b": 30.5, "act_params_b": 3.3,
        "tile_default": [512, 1024, 768], "provenance": "public",
    },
    "qwen3-32b": {
        "family": "qwen3", "label": "Qwen3-32B（dense）", "moe": False,
        "layers": 64, "num_experts": 0, "top_k": 0, "hidden": 5120, "mlp": 25600,
        "vocab": 151936, "params_b": 32.8, "act_params_b": 32.8,
        "provenance": "public",
    },
    "llama3.1-405b": {
        "family": "llama", "label": "Llama 3.1 405B（dense）", "moe": False,
        "layers": 126, "num_experts": 0, "top_k": 0, "hidden": 16384, "mlp": 53248,
        "vocab": 128256, "params_b": 405.0, "act_params_b": 405.0,
        "provenance": "public",
    },
    "llama3.1-70b": {
        "family": "llama", "label": "Llama 3.1 70B（dense）", "moe": False,
        "layers": 80, "num_experts": 0, "top_k": 0, "hidden": 8192, "mlp": 28672,
        "vocab": 128256, "params_b": 70.6, "act_params_b": 70.6,
        "provenance": "public",
    },
    "mixtral-8x22b": {
        "family": "mixtral", "label": "Mixtral 8x22B", "moe": True,
        "layers": 56, "num_experts": 8, "top_k": 2, "hidden": 6144, "mlp": 16384,
        "vocab": 32768, "params_b": 141.0, "act_params_b": 39.0,
        "tile_default": [512, 2048, 4096], "provenance": "public",
        "note": "只有 8 个专家 —— 专家维分片几乎必然除不尽，别往那边想。",
    },
}

FAMILY_LABEL = {
    "hunyuan3": "Hunyuan3", "deepseek": "DeepSeek", "qwen3": "Qwen3",
    "llama": "Llama", "mixtral": "Mixtral",
}


# ── MoE 后端 ────────────────────────────────────────────────────
# 用户选的是「哪条 kernel 路径」，不是一串 flag 名字。
# apply 里是这条路径真正对应的参数，UI 负责把它们一次性写进配置。
BACKENDS: dict[str, dict] = {
    "native": {
        "label": "native megablox",
        "desc": "自带 Pallas kernel，权重收集由编译器按分片规格插入。**默认选它。**",
        "apply": {"megablox": True, "sparse_matmul": True, "use_tokamax_gmm": False},
        "pros": ["通信能被藏住：每步暴露 34.6 ms（tokamax 是 6,170 ms）",
                 "80 层的收集合并、提升出循环，每步只发一次"],
        "cons": ["⚠️ 配专家维分片时会**静默漏算** —— 所以别开专家维分片"],
    },
    "tokamax": {
        "label": "tokamax",
        "desc": "同一个开关在两个精度下走**完全不同的代码**："
                "BF16 是裸 `ragged_dot`，FP8 才是 megablox + tokamax 后端。",
        "apply": {"megablox": True, "sparse_matmul": True, "use_tokamax_gmm": True},
        "pros": ["FP8 下支持跨卡量化收集，收的是量化后权重，字节减半",
                 "配专家维分片时不会漏算（收集是手写在 kernel 入口的）"],
        "cons": ["手写集合通信钉在依赖链中间，调度器藏不住 —— 暴露耗时是 native 的 178 倍",
                 "BF16 走裸路径，实测慢 12 倍，没有任何理由用",
                 "不设 Mosaic 参数时 kernel 只跑出峰值的 0.67%，且不报错"],
    },
    "dense_matmul": {
        "label": "dense matmul（token dropping）",
        "desc": "不走分组矩阵乘，超容量的 token 直接丢。",
        "apply": {"megablox": False, "sparse_matmul": False, "use_tokamax_gmm": False},
        "pros": ["实现简单，容量固定，显存好估"],
        "cons": ["丢 token 影响收敛质量", "大专家数下算力浪费严重"],
    },
}


def detect_backend(params: dict) -> str:
    if params.get("use_tokamax_gmm"):
        return "tokamax"
    if params.get("sparse_matmul") is False or params.get("megablox") is False:
        return "dense_matmul"
    if params.get("megablox") or params.get("sparse_matmul"):
        return "native"
    return ""


def effective_shape(params: dict) -> dict:
    """注册表形状 + **用户显式覆盖**，合并成这次实际要编译的形状。

    改层数是很常见的操作（调配置时用 4–8 层跑得快 3 倍），但
    **参数量、常驻显存、能不能装下，全都随层数变** —— 如果这里只读注册表，
    工具就会拿生产层数的结论去回答一个减层配置的问题，而且一声不吭。
    这跟「配置在语义上错了却不报错」是同一族错误。
    """
    m = get_model(params.get("model_name"))
    if not m:
        return {}
    out = dict(m)
    prod_layers = m["layers"]
    try:
        L = int(params.get("num_decoder_layers", prod_layers) or prod_layers)
    except (TypeError, ValueError):
        L = prod_layers
    out["prod_layers"] = prod_layers
    out["layers"] = L
    out["layers_overridden"] = L != prod_layers

    # 嵌入层不随层数变，要先扣出来再按层折算 —— 直接按比例缩会把嵌入也缩掉
    emb_b = (m["vocab"] * m["hidden"] * 2) / 1e9
    per_layer = max(m["params_b"] - emb_b, 0) / prod_layers
    out["emb_params_b"] = round(emb_b, 3)
    out["params_b"] = round(emb_b + per_layer * L, 3)
    act_per_layer = max(m["act_params_b"] - emb_b, 0) / prod_layers
    out["act_params_b"] = round(emb_b + act_per_layer * L, 3)
    out["layer_ratio"] = L / prod_layers if prod_layers else 1.0
    return out


def get_model(name) -> dict:
    return MODELS.get(str(name or "").lower(), {})


def catalog() -> dict:
    """给前端的下拉数据：按族分组。"""
    out: dict[str, list] = {}
    for k, v in MODELS.items():
        out.setdefault(v["family"], []).append({
            "id": k, "label": v["label"], "moe": v["moe"],
            "layers": v["layers"], "num_experts": v["num_experts"],
            "hidden": v["hidden"], "mlp": v["mlp"], "params_b": v["params_b"],
            "provenance": v["provenance"], "note": v.get("note", ""),
            "tile_default": v.get("tile_default"),
        })
    return {"families": [{"id": f, "label": FAMILY_LABEL.get(f, f), "models": m}
                         for f, m in out.items()],
            "backends": [{"id": k, **v} for k, v in BACKENDS.items()]}
