# -*- coding: utf-8 -*-
"""专题二自己的图 —— 转发显微镜那套绘图库，不再抄第三份。

gpu-micro 和 tpu-micro 各有一份逐字节相同的 common.py（移植脚本里有断言兜着）。
再抄第三份，等于给「改一个常数要记得改三个地方」埋雷 ——
gpu-micro 那份的代码字宽系数就是这么跟 tpu-micro 走散到 0.92 的，
表现为「代码多的段落冲出卡片边框」，排查花的时间比当初省下的多得多。

所以这里只做转发：把 tpu-micro/common.py 当成唯一真源加载进来，
图文件照常写 `from common import Fig, para, ...`，不用关心它其实在隔壁。
"""
import importlib.util
import os

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "tpu-micro", "common.py")

_spec = importlib.util.spec_from_file_location("_microscope_common", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
