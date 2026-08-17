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

    out.update(scale_params(m, L))
    out["layer_ratio"] = L / prod_layers if prod_layers else 1.0
    return out


def scale_params(m: dict, L: int, vocab=None, hidden=None) -> dict:
    """按层数折算参数量。**嵌入层不随层数变，要先扣出来** ——
    直接按比例缩会把嵌入也一起缩掉，小层数配置的参数量会明显偏小。

    这一份是**唯一实现**：会话里改 `num_decoder_layers` 走它，
    模型配置面板里另存新模型也走它。两处各算一遍迟早对不上，
    而对不上的表现是「40 层的模型显示 297B 参数」—— 看着像没生效。
    """
    prod = m["layers"] or 1
    v = int(vocab if vocab is not None else m["vocab"])
    h = int(hidden if hidden is not None else m["hidden"])
    emb_b = (v * h * 2) / 1e9
    per_layer = max(m["params_b"] - emb_b, 0) / prod
    act_per_layer = max(m["act_params_b"] - emb_b, 0) / prod
    return {"emb_params_b": round(emb_b, 3),
            "params_b": round(emb_b + per_layer * L, 3),
            "act_params_b": round(emb_b + act_per_layer * L, 3)}


# ── 自定义模型 ──────────────────────────────────────────────────
# 内置那几个是**只读**的：它们的形状对应真实发布的模型，改了以后
# 「Hunyuan3-295B」这个名字就不再指那个模型了，而所有实测结论都是挂在
# 这个名字上的。要改层数 / 专家数就另存一个名字，血缘留在 `derived_from`。
CUSTOM: dict[str, dict] = {}          # 由 app.py 在启动时从存储灌进来

# 允许改的形状字段（值类型 → 前端据此渲染）
EDITABLE = [
    {"k": "layers", "label": "层数", "type": "int",
     "why": "最常改的一个。层数直接决定参数量、显存和编译时长。"},
    {"k": "num_experts", "label": "专家数", "type": "int",
     "why": "MoE 才有。要能被 FSDP 宽度整除，否则专家维分片开不了。"},
    {"k": "top_k", "label": "top-k", "type": "int", "why": "每个 token 激活几个专家。"},
    {"k": "hidden", "label": "hidden", "type": "int", "why": "tile 的 embed 维上界看它。"},
    {"k": "mlp", "label": "mlp", "type": "int", "why": "MoE 单专家的 mlp 维，tile 的 n 维上界。"},
    {"k": "vocab", "label": "词表", "type": "int",
     "why": "logits 显存 = batch × seq × 词表，大词表模型这一块常占 peak 的一多半。"},
    {"k": "params_b", "label": "总参数(B)", "type": "float", "derived": True,
     "why": "显存估算用。**改层数会自动折算**（嵌入层先扣出来），自己填了就以你填的为准。"},
    {"k": "act_params_b", "label": "激活参数(B)", "type": "float", "derived": True,
     "why": "MoE 的实际计算量。同样随层数自动折算。"},
]


def all_models() -> dict:
    """内置 + 自定义。**自定义不能覆盖内置** —— 同名时内置赢。"""
    out = dict(CUSTOM)
    out.update(MODELS)
    return out


def is_builtin(name) -> bool:
    return str(name or "").lower() in MODELS


def get_model(name) -> dict:
    return all_models().get(str(name or "").lower(), {})


def catalog() -> dict:
    """给前端的下拉数据：按族分组。"""
    out: dict[str, list] = {}
    for k, v in all_models().items():
        out.setdefault(v["family"], []).append({
            "id": k, "label": v["label"], "moe": v["moe"],
            "layers": v["layers"], "num_experts": v["num_experts"],
            "hidden": v["hidden"], "mlp": v["mlp"], "params_b": v["params_b"],
            "provenance": v["provenance"], "note": v.get("note", ""),
            "tile_default": v.get("tile_default"),
            "top_k": v.get("top_k"), "vocab": v.get("vocab"),
            "act_params_b": v.get("act_params_b"),
            "builtin": k in MODELS, "derived_from": v.get("derived_from"),
        })
    return {"families": [{"id": f, "label": FAMILY_LABEL.get(f, f), "models": m}
                         for f, m in out.items()],
            "backends": [{"id": k, **v} for k, v in BACKENDS.items()],
            "editable": EDITABLE}
