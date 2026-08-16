# backend —— FastAPI

```bash
cd ~/gpu-tpu-pedia/tpu
python3 -m uvicorn tpuguru.backend.app:app --host 127.0.0.1 --port 8820 --reload
```

| 文件 | 干什么 |
|---|---|
| `app.py` | 路由、会话、存档、配置指纹、BotCall 代理 |
| `parser.py` | 命令解析 + `train` → `train_compile` 转换 + 回环校验 |
| `lint.py` | 规则引擎（数据驱动，规则在 `rules/rules.seed.json`） |
| `worker.py` | AOT 执行器：`real`（docker）/ `replay`（回放实测结论） |

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `TPUGURU_AOT_PROFILE` | 空 | **设了才走 real 模式**，指向执行档案 JSON（见 `worker/aot-profile.hy3.json`） |
| `TPUGURU_BOT_URL` | `http://127.0.0.1:8810/api/chat` | BotCall 目标（tpuguru bot 的 web channel） |
| `TPUGURU_FORCE_LOCAL` | 空 | `1` 强制用本地 JSON 存储，不连 Firestore |
| `TPUGURU_LOCAL_DIR` | `/tmp/tpuguru-store` | 本地存储目录 |

## real 模式怎么开

```bash
TPUGURU_AOT_PROFILE=$PWD/worker/aot-profile.hy3.json \
  python3 -m uvicorn tpuguru.backend.app:app --port 8820
# /api/health 里 aot_mode 会变成 real
```

执行档案里最要紧的是 `param_map`：工具里的参数名跟镜像里那套代码**不一定同名**。
`num_decoder_layers` 在这套 MaxText 里叫 `base_num_decoder_layers` ——
名字不对会被 `base.yml` **静默忽略**，你以为在跑 40 层，其实跑的是生产层数。
又是一次「配置改了但没生效」，所以映射必须显式写出来。

`_parse_aot_output` 要认**三种**失败，含义完全不同：

| 报错 | 含义 | 降 batch 有用吗 |
|---|---|---|
| `HLO temporaries (X G) exceeds available HBM` | 运行期临时缓冲超 | 通常有效 |
| `CompileTimeHbmOom ... Exceeded hbm capacity by X` | 连排布方案都找不到 | 未必线性 |
| `CompileTimeScopedVmemOom` | VMEM 不够，不是 HBM | **基本没用**，要降 tile / block size |

## 踩过的坑

- **`uvicorn` 不带 `--reload` 改了 Python 不生效。** 前端是静态文件、改了立刻见效，
  于是很容易出现「前端变了、后端没变」，把人引到错误的结论上。开发一律带 `--reload`。
- **`"1"` 不能当布尔。** `ici_expert_parallelism=1` 是**并行度 1**；一旦被 coerce 成
  `True`，FSDP 宽度就算错，后面整条 lint 链跟着错，而且错得很像对的。
  只有 `true/false/yes/no/on/off` 才是布尔。
- **规则求值失败必须出声。** 一条不生效的 lint 跟根本没有这条规则，用户是看不出区别的。
  引擎会把失败的规则汇总成一条 `ENGINE` info 顶到结果里。
- **`run_aot` 差点被自己切掉。** 用「按函数名切片再拼」的方式改文件时，
  中间夹着的函数会被一起吃掉，而且是 import 时才报错。改完一定要先起一次服务。
- **报告要带配置指纹。** 跑完之后改配置，旧报告还挂在那儿 ——
  没有指纹对比，用户会对着上一次的结论调这一次的参数。
