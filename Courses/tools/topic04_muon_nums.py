# -*- coding: utf-8 -*-
r"""专题四 · Muon 那几张图共用的数（2026-11-08）。

⭐ 一处现算、几张图共用：fig4-muonstep、fig4-ns、fig4-keeplose、fig4-pot
   都用「三个频率，强度 10 ／ 1 ／ 0.01」这个例子。数抄进各图会漂，所以集中在这里算、在这里断言。

📌 出处（已读原文／原码）：
  · Newton-Schulz 五次多项式系数 (a, b, c) ＝ (3.4445, −4.7750, 2.0315)、跑 5 轮、
    先除以 Frobenius 范数（「总大小」）、动量 0.95、Nesterov、形状系数 √max(1, 行/列)：
    Keller Jordan 的博客 kellerjordan.github.io/posts/muon 与公开实现 github.com/KellerJordan/Muon（muon.py）
  · 经典三次迭代 1.5x − 0.5x³（Kovarik 1970），Bernstein《Deriving Muon》里作为最简单的例子
  · 「奇数次矩阵多项式只作用在奇异值上、奇异向量不动」：同上两篇
"""
import numpy as np

A, B, C = 3.4445, -4.7750, 2.0315
STEPS = 5
BETA = 0.95


def p5(x):
    """Muon 的五次多项式：作用在每个强度（奇异值）上。"""
    return A * x + B * x ** 3 + C * x ** 5


def p3(x):
    """经典三次：1.5x − 0.5x³。"""
    return 1.5 * x - 0.5 * x ** 3


def iterate(fn, x, n=STEPS):
    out = [np.array(x, dtype=float)]
    for _ in range(n):
        out.append(fn(out[-1]))
    return out


# ── 三个频率的小例子 ───────────────────────────────────────────────
RAW = np.array([10.0, 1.0, 0.01])
SCALED = RAW / np.sqrt((RAW ** 2).sum())              # 除以总大小（Frobenius 范数 ＝ 强度平方和开根号）
TRACE = iterate(p5, SCALED)                            # 第 0..5 轮
FINAL = TRACE[-1]
E_BEFORE = RAW ** 2 / (RAW ** 2).sum()
E_AFTER = FINAL ** 2 / (FINAL ** 2).sum()
CLASSIC = iterate(p3, SCALED)[-1]

assert np.allclose(np.round(SCALED, 4), [0.995, 0.0995, 0.001], atol=1e-4), SCALED
assert np.allclose(np.round(FINAL, 2), [0.70, 0.71, 0.47]), FINAL
assert np.allclose(np.round(E_BEFORE * 100), [99, 1, 0]), E_BEFORE
assert np.allclose(np.round(E_AFTER * 100), [41, 41, 18]), E_AFTER
assert FINAL[1] > FINAL[0], "排名被轻微打乱：原来最强的那个，五轮后反而略小 —— 图上写了这句"
assert np.all(FINAL > 0), "不会反向：五轮后每个强度都还是正的"
# 小数每轮放大三倍多：p5 在 0 附近的斜率就是 a
assert 3.4 < A < 3.5 and abs(TRACE[1][2] / TRACE[0][2] - A) < 0.01
# 经典三次五轮：0.001 那个几乎没动（慢）
assert CLASSIC[2] < 0.01 and CLASSIC[1] < 0.7, CLASSIC

# ── 多项式本身的性质 ───────────────────────────────────────────────
_xs = np.linspace(1e-6, 1.0, 200001)
_ys = iterate(p5, _xs)[-1]
LO_1E3 = _ys[_xs >= 1e-3].min()                        # 起点 ≥ 0.001 的，五轮后最低多少
HI = _ys.max()
assert 0.47 < LO_1E3 < 0.48 and 1.20 < HI < 1.21, (LO_1E3, HI)
_ys50 = iterate(p5, np.linspace(0.3, 1.2, 2001), 50)[-1]
OSC_LO, OSC_HI = _ys50.min(), _ys50.max()              # 多跑几十轮也只在这个区间里晃
assert 0.68 < OSC_LO < 0.69 and 1.13 < OSC_HI < 1.14, (OSC_LO, OSC_HI)
_r = np.roots([C, 0, B, 0, A - 1, 0])
BLOW = max(r.real for r in _r if abs(r.imag) < 1e-9)   # 超过这个数，就被越推越大
assert 1.26 < BLOW < 1.27, BLOW
with np.errstate(over="ignore", invalid="ignore"):
    assert abs(iterate(p5, np.array([1.3]))[-1][0]) > 1e50       # 1.3 五轮就炸
    assert abs(iterate(p5, np.array([1.25]))[-1][0]) < 1.3        # 1.25 还没炸

# ── 「只改强度，不动模式」：XXᵀX ＝ U Σ³ Vᵀ（拿一块随机矩阵验一遍） ─────────────
_rng = np.random.default_rng(0)
_X = _rng.normal(size=(5, 3))
_u, _s, _vt = np.linalg.svd(_X, full_matrices=False)
assert np.allclose(_X @ _X.T @ _X, _u @ np.diag(_s ** 3) @ _vt)
_Xn = _X / np.linalg.norm(_X)
_Y = _Xn.copy()
for _ in range(STEPS):
    _AA = _Y @ _Y.T
    _Y = A * _Y + (B * _AA + C * _AA @ _AA) @ _Y          # 公开实现里的写法，逐字照抄
# ⭐ 五轮以后，输出 ＝ 同一组输入／输出模式 × 每个强度各自过一遍多项式（模式一个没动）
assert np.allclose(_Y, _u @ np.diag(iterate(p5, _s / np.linalg.norm(_X))[-1]) @ _vt)

# ── 不会反向：更新与动量的内积 ＝ Σ σᵢ·p(σᵢ) > 0 ────────────────────────
_sn = _s / np.linalg.norm(_X)
assert np.isclose(np.sum(_Xn * _Y), np.sum(_sn * iterate(p5, _sn)[-1]))   # 内积只看各频率强度的乘积
assert np.sum(_Xn * _Y) > 0

# ── [[2,1],[1,2]] → 单位矩阵：行变了，频率方向没变 ───────────────────────
M22 = np.array([[2.0, 1.0], [1.0, 2.0]])
_u3, _s3, _vt3 = np.linalg.svd(M22)
O22 = _u3 @ _vt3
assert np.allclose(O22, np.eye(2))
_Z = M22 / np.linalg.norm(M22)
for _ in range(40):
    _Z = 1.5 * _Z - 0.5 * _Z @ _Z.T @ _Z                    # 经典迭代跑够轮数，收敛到正交矩阵
assert np.allclose(_Z, np.eye(2), atol=1e-9)

# ── 形状系数 ─────────────────────────────────────────────────────
SHAPE = (6144, 2048)
SHAPE_K = max(1.0, SHAPE[0] / SHAPE[1]) ** 0.5
assert abs(SHAPE_K - 3 ** 0.5) < 1e-12
