# Remote CLIP Colab 运行时 — 需求文档（v1.2，已确认）

> 状态：**已实施并验证**（2026-08-16 实施验证；2026-08-17 v1.2 引入 native 后端）。
> v1.2 变更：按用户决策改走 **vendor 路线**重写 text encoder 支持——
> `colab/comfy/`（GPL-3，`vendor_comfy.ps1` 同步）+ `engines_native.py` 桥接
> `comfy.sd.load_text_encoder_state_dicts`，worker `--engine auto|native|hf`
> （auto 子进程探测；TPU/CPU-only 强制 hf）。
> 效果：ComfyUI 支持的全部 text encoder 格式（37 种 TEModel 检测、内嵌
> tokenizer、Long-CLIP、Qwen3 全家、NVFP4/AWQ 量化）在远程端**构造性等价**可用；
> LoRA 走 comfy 本体 patcher（克隆链 + LRU(4)）。本地节点新增 `sources`/
> `clip_type` 输入。
> 验证（RTX 4060 Ti, native）：clip_l/sdxl/flux 形状正确、LoRA 注入/回退干净、
> **MiniMax H3 qwen3vl-32b NVFP4 加载 3s + 编码 19s（含 token_tags 输出）**；
> hf 回退（CPU-only torch）与共存正常。
> **真实 Colab 实测（2026-08-17，google-colab-cli）**：
> - T4 GPU（native 后端 + Cloudflare 隧道）：本机→公网全链路 9/9 PASS
>   （TEST_REPORT_T4.md）；flux encode 3.4 s。
> - TPU v5e（hf 后端，bf16 + 形状分桶）：9/9 PASS（TEST_REPORT_TPU.md）；
>   首次 encode 含 XLA 编译 9.9 s，同 bucket 复用 1.4 s。客户端 encode 超时
>   因此从 120 s 提升至 600 s（ENCODE_TIMEOUT）。notebook cell 1 在 TPU 上
>   跳过 torch/comfy-kitchen 安装（保护镜像预装的 torch↔torch_xla 配对）。
>
> **推理加速（2026-08-17 第二轮）**：
> - GPU native：`--attention auto|sdpa|sage|flash` 接入 comfy 本体注意力分派
>   （import 前注入 cli_args，engines_native 惰性导入重构）。实测（4060 Ti）：
>   文本编码器走 small-input 路径，NVIDIA+torch≥2 已默认 SDPA 融合核，
>   三模式输出逐位一致（rel 0.0）——与上游 comfy 行为一致，sage/flash 对
>   小 token 数文本编码器无收益（属扩散模型大序列优化）。
> - GPU hf：`--attention-hf sdpa|eager|flash_attention_2` 传入 from_pretrained。
> - TPU：`--xla-cache DIR` 持久化 XLA 编译产物。真机 v5e 对照：跨 worker
>   重启首请求 **11.3 s → 8.4 s**（消除重编译；剩余为 ~10 GB 权重装载），
>   热路径 ~2 s 不变。notebook 新增 ATTENTION/ATTENTION_HF/XLA_CACHE 表单项。
>
> **审查修复（2026-08-18，全项目 review，7 项）**：
> - **hf 引擎 LoRA 泄漏（高危）**：encode/generate 空栈不再跳过
>   `LoraPatcher.apply`（apply 兼任回退路径）；set_default/unload/同名重载经
>   `_retire_engine` 重置 patcher（回退权重 + 释放 pristine CPU 副本）。
>   回归实测 11/11（伪造引擎 + kohya LoRA：注入→空栈回退→换默认→卸载→generate 路径）。
> - `probe_native` 子进程 cwd 固定为脚本目录：README 根目录直启 `python colab/worker.py`
>   不再静默降级 hf（本机差分验证：根 cwd `No module named 'comfy'`，colab cwd 可导入）。
> - 客户端网络层错误（URLError/Timeout/连接中断）统一包装为 WorkerError；
>   503 `{"error":{...}}` 体提取可读 message；generate 轮询容忍 5 次连续瞬时失败
>   （404 立即失败），超时后的 DELETE 清理失败不再掩盖超时错误——stub server 实测 8/8。
> - `shutdown` 控制面新增 `shutdown_confirm` 布尔输入（默认 False，误排队不再杀 worker）；
>   load_model 空 source 不下发（服务端 422 如期触发；空 components 同样 422）；
>   native `_resolve_source` 裸文件名（带/不带扩展名）正确搜索 models 目录（9/9 用例）。
> - notebook 删除 cell 5 ipywidgets 控制面板（实际使用不可用）——管理操作全部经由
>   ComfyUI `Remote CLIP Controller` 节点；cell 6/7 顺次改为 5/6，README 与
>   notebook 内提示同步更新。
> 原确认结论：Q1 Phase 1+2 合并；传输固定 fp16；TPU v5e 走 hf 后端
>（bf16 计算 + 形状分桶，待 Colab 实测）。
> 原型：`custom_nodes/ComfyUI-RemoteCLIPLoader`（v1.2.2，下称"旧版"）。
>
> **验证记录**（本机 CPU fp32 + fp16 传输实测，参照 comfy 本体输出）：
> - SD1 clip_l：cond/pooled 与 comfy 逐位对齐（线上 rel ~2.5e-4，纯 fp16 传输噪声）
> - SDXL 组合 + kohya LoRA（0.8 强度）注入/还原：本地 **逐位一致（rel 0.0）**，线上一致
> - T5-XXL(24L)：rel 9.3e-6；Flux 组合：cond 9.3e-6、pooled **精确 0.0**，线上 3.1e-4
> - SD3 三编码器组合：cond rel 1.1e-7、pooled 精确 0.0（含 LoRA 回归无损）
> - REST 控制面全动作、鉴权 401、协议不符 409、无模型 400、Job 202/轮询/删除/404、
>   LoRA 上传/删除/跨角色过滤（clip_g LoRA 不误伤 clip_l 引擎）
> - Cloudflare quick tunnel：URL 实时提取，公网 HTTPS 全链路（协议协商/远程加载/encode）实测通过

---

## 1. 背景与目标

旧版将 CLIP 文本编码器（含生成式编码器，如 Gemma）卸载到另一台机器运行，但其
**Sender（推理端）是一个 ComfyUI 节点**——必须在完整 ComfyUI 环境中、通过排队工作流才能启动，
且使用 raw TCP 私有协议，仅适合局域网。

本项目目标：

| # | 目标 | 说明 |
|---|------|------|
| G1 | 推理端完全独立 | 远程推理脚本**零 ComfyUI 依赖**，可直接在 Google Colab（免费 T4 即可）一个单元内启动 |
| G2 | 通信全面 RESTful 化 | 本地加载侧通过一套 **RESTful API** 完成对远程端的**推理 + 控制**（模型加载/卸载、LoRA 管理、缓存、状态、长任务） |
| G3 | 内置隧道 | 远程端自带 **Cloudflare Tunnel 式**内网穿透，自动获得公网 HTTPS 地址，无需端口转发/公网 IP |
| G4 | 接收端保留完整功能 | 旧版 Sender 的全部能力（encode / generate / LoRA 转发 / 缓存 / 鉴权 / 传输精度控制）在新运行时中等价保留 |
| G5 | 本地侧为 ComfyUI 节点 | 对工作流呈现为普通 CLIP 对象，`CLIPTextEncode`、`Generate Text` 等节点即插即用 |

**非目标（Phase 1 不做）**

- 不做多人多租户调度（一机一模型服务即可，支持多模型常驻切换）。
- 不在核心 ComfyUI 侧新增任何对外互联网请求路径（本地侧仅访问用户自行配置的远程地址）。
- 不承诺与旧版 v2 TCP 协议互通（见 §10 待确认 Q5）。

---

## 2. 现状分析（旧版）

旧版单文件 `__init__.py`，三个节点：

- `SendRemoteCLIP`：**在 ComfyUI 内**起 raw TCP server（协议 v2：8 字节大端长度前缀 + JSON header + 二进制 blob；tensor manifest 打包；HMAC 比对 token）。依赖 `comfy.sd` / `comfy.utils` / `folder_paths` 加载与补丁 CLIP。
- `LoadRemoteCLIP`：返回 `RemoteCLIPProxy`，鸭子类型实现 `tokenize / encode_from_tokens / encode_from_tokens_scheduled / generate / decode / clone / with_lora`。
- `LoraLoaderCLIPOnly`：检测到远程代理时只转发 `(lora_name, strength)`，由 Sender 侧查找并应用。

已验证有价值的机制（新版本全部继承）：

1. tensor manifest 打包格式（dtype/orig_dtype/shape/offset/size + 原始字节流），dtype 白名单防解码攻击；
2. fp16 传输降带宽（auto 模式：本地 fp32 / 远程 fp16）；
3. 双层 LRU 缓存：embedding 缓存（64）+ LoRA 补丁后模型缓存（4）；
4. generate 不重试（避免重复昂贵的自回归采样）、encode 自动重连重试；
5. 协议版本不匹配时显式拒绝。

旧版缺陷（新版解决）：

| 缺陷 | 新版对策 |
|------|----------|
| Sender 依赖完整 ComfyUI 运行环境 | 独立 Python 运行时，Colab 一个 cell 启动 |
| raw TCP 明文、仅局域网 | HTTPS（隧道端到 TLS）+ Bearer token，公网可用 |
| 无法远程管理（换模型要重排队工作流） | RESTful 控制面：模型/LoRA/缓存/状态/关机 |
| Colab 无公网入站，旧版根本跑不了 | 内置 cloudflared 隧道 |
| TCP 长连接断线即任务失败 | HTTP 无状态 + 长任务 Job 化（断线轮询恢复） |

---

## 3. 总体架构

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│ 本地机（跑 ComfyUI 工作流） │         │  Google Colab / 任意 Python 机 │
│                         │         │                              │
│  工作流节点               │         │  remote-clip-worker（独立进程） │
│  ├─ CLIPTextEncode ──────┼──HTTPS──┼─▶ FastAPI :8188 (127.0.0.1)  │
│  ├─ Generate Text ───────┼─ REST   │    ├─ /v1/encode  /v1/generate│
│  └─ LoadRemoteCLIPColab ▶│  v3     │    ├─ /v1/models/* /v1/loras*│
│       (CLIP 代理对象)     │         │    └─ /v1/status /v1/jobs/*  │
│  RemoteCLIPController ▶─┼─────────┼─▶ 引擎层：CLIP/T5/CausalLM     │
│       (管理/诊断, 可选)    │         │    + LoRA 引擎 + 双层缓存      │
└─────────────────────────┘         │  cloudflared（自动拉起，出站连接）│
        ▲                            └──────────────┬───────────────┘
        └──────── https://xxxx.trycloudflare.com ◀──┘（打印到 Colab 输出）
```

- **控制面与数据面统一走 HTTP**（同一端口、同一鉴权），本地侧用 Python 标准库 `urllib`/`http.client` 直连或经隧道访问，无状态、可任意重连。
- 大张量载荷不放进 JSON：定义 `application/x-rcp-v3` 打包体（见 §5.4），沿用旧版 manifest 思路。

---

## 4. 远程推理端（Colab 运行时）

### 4.1 形态与依赖

- 一个可整目录拷贝/克隆的独立 Python 包 `colab/`，入口 CLI：
  ```bash
  python worker.py --model "openai/clip-vit-large-patch14" --tunnel cloudflare --host 127.0.0.1 --port 8188
  ```
  Colab 中即三条命令：克隆 → `pip install -r colab/requirements.txt` → 启动（README 提供 cell 模板）。
- 依赖（仅 Colab 侧，本地侧零新增）：`torch`、`transformers`、`safetensors`、`fastapi`、`uvicorn`、`numpy`、`huggingface_hub`。
- 服务器仅绑定 `127.0.0.1`，**所有外部流量必须经隧道**，由隧道提供 TLS。

### 4.2 模型引擎（Phase 1 范围，待确认 Q1）

| kind | 架构 | 用途 | 典型模型（Colab 显存参考） |
|------|------|------|---------------------------|
| `clip` | HF `CLIPTextModel` / OpenCLIP bigG | SD1.x / SD2 / SDXL 单编码器条件编码 | clip-l（~0.5GB bf16）、clip-g（~2.5GB bf16，T4 可用） |
| `clip_composite` | clip-l + clip-g 组合（双路编码，拼 cond、取 g 路 pooled） | SDXL | 同上双模型 |
| `t5` | HF `T5EncoderModel` | 单独 T5 编码 | t5-xxl bf16（~9.5GB，T4 勉强 / L4 舒适） |
| `flux` | clip-l + t5-xxl 双编码器（clip 路 zero-pad 到 t5 形状后拼接） | Flux.1 | clip-l + t5-xxl（~10GB bf16） |
| `sd3` | clip-l + clip-g + t5-xxl 三编码器（各自投影后拼接） | SD3.5 | 三模型（~12GB bf16，L4/A100 舒适） |
| `qwen_image` | Qwen2.5-VL 语言模型路 | Qwen-Image / Qwen-Image-Edit | qwen2.5-vl-7b bf16（~16GB，L4/A100） |
| `causal_lm` | HF `AutoModelForCausalLM` | 生成式编码：条件编码（末 token / 均值池化）+ 文本生成 | gemma-3-4b-it（T4 可跑）、qwen3-4b 等 |

- 模型来源两种：HF repo id（由用户显式指定才下载，遵守"用户显式发起才允许下载模型"原则）或 Colab 本地路径（safetensors）。
- 加载参数：`dtype`（auto→GPU 选 bf16，TPU 选 bf16，CPU 选 fp32）、`device`（auto/cuda/tpu/cpu）。
- **多模型常驻**：可 `POST /v1/models/load` 加载多个，指定一个 `default` 服务 encode/generate；显存不足时加载报明确错误（LRU 自动卸载未实现，列为后续项）。

**TPU v5e 适配**（评估结论已确认可行）：
- 计算 dtype 固定 bf16（TPU 不支持 fp16 计算；传输仍为 fp16，转换发生在 CPU 侧）。
- 设备经 `xm.xla_device()` 获取，`torch_xla` 可选依赖探测式导入，收尾 `xm.mark_step()`。
- 动态长度模型（t5 / causal_lm / qwen_image）输入 padding 到固定形状桶（77/128/256/512），限制 XLA 重编译次数；启动时预热常用桶。
- 单进程单线程推理（沿用 `_infer_lock` 串行），不做多芯 SPMD。
- 已知限制：generate 自回归在 XLA 上显著慢于 GPU（功能正确，速度列为已知项）；bitsandbytes 量化不可用，大模型量化放后续。

### 4.3 LoRA 引擎

- 支持 kohya 格式 CLIP/T5/LLM 文本编码器 LoRA（safetensors），自实现 ~百行的 `lora_up/lora_down + alpha` 权重补丁注入（按模块名映射到 HF 层），不引入 peft。
- LoRA 栈按请求随 `lora_stack: [[name, strength], ...]` 传入，与旧版语义一致：强度 0 跳过、栈相同走补丁缓存（LRU 4）、文件不存在返回明确 404 错误。
- LoRA 文件进 Colab 两条路：手动上传到 `models/loras/`，或 `PUT /v1/loras/{name}`（带 token，safetensors 白名单校验）。

### 4.4 缓存（等价继承旧版）

- embedding LRU（默认 64，key = 模型 + text + kwargs + lora_stack 的规范化哈希；含图像/视频等多模态张量输入的请求不缓存）。
- LoRA 补丁模型 LRU（4）。
- `POST /v1/cache/clear` 一键清空；`/v1/status` 报告命中数。

### 4.5 内置隧道（Q2）

| 模式 | 说明 | 默认 |
|------|------|------|
| `cloudflare` | 自动下载 `cloudflared` 二进制 → 拉起 quick tunnel（免账号、免配置）→ 解析并**醒目打印** `https://*.trycloudflare.com` | ✅ |
| `ngrok` | 用户已有 ngrok token 时可选（`--tunnel ngrok --ngrok-token ...`） | 可选 |
| `direct` | 绑定 `0.0.0.0` 直连（VPS/局域网，绕过隧道的体积/时长限制） | 可选 |

（localtunnel 已从范围移除：cloudflare + ngrok + direct 已覆盖需求场景。）

- 隧道进程由 worker 托管：崩溃自动重启、URL 变化重打印；`GET /v1/tunnel` 随时查当前公网 URL 与状态。
- 隧道模式下**强制要求 token**：未提供则自动生成随机 token 并与 URL 一起打印（quick tunnel URL 是公开的，鉴权必须开）。
- 已知隧道限制及对策：
  - Cloudflare 免费层单请求体积上限（~100MB）→ 超大视频输入建议 `direct` 模式；
  - Cloudflare 边缘对首字节响应超时（~100s）→ **generate 走异步 Job**（§5.3），encode 同步但应在限时内完成，超时返回可重试错误。

### 4.6 Colab 环境适配

- 启动时探测 GPU 型号/显存，打印建议 dtype；无 GPU 时允许 CPU 模式并警告。
- 单 cell 全自动：依赖安装失败给出清晰错误；cloudflared 下载支持 Linux x86_64/arm64。
- 不做反空闲保活（避免违反 Colab 使用条款）；README 说明 T4 免费档单次时长与断连行为，HTTP 无状态设计保证 Colab 重启后只需把新 URL/token 填回本地节点。
- 显存指引表（README）：T4 16GB / L4 24GB / A100 40GB 对应推荐模型与量化档位。

---

## 5. RESTful API 规范（协议 v3）

### 5.1 通用约定

- Base：`http(s)://<host>:<port>`，所有路径前缀 `/v1`。
- 鉴权：`Authorization: Bearer <token>`（除 `/health`），错误统一
  `{"error": {"code": "unauthorized", "message": "..."}}`，HTTP 状态码：400/401/404/409/413/422/500/503。
- 版本协商：`GET /v1/status` 返回 `proto: 3` 与能力集，本地节点版本不符时拒绝并给出明确提示（继承旧版"协议不匹配显式拒绝"原则）。
- `/docs`（FastAPI 自动生成的 OpenAPI 文档）可直接在浏览器里调试整套控制面。

### 5.2 控制面端点

| 方法 | 路径 | 请求 | 响应 |
|------|------|------|------|
| GET | `/health` | — | `{"ok": true}` |
| GET | `/v1/status` | — | 版本/协议/GPU/显存/已加载模型/默认模型/缓存命中/隧道 URL/运行时长 |
| GET | `/v1/models` | — | 本地已有模型文件（相对 `--models-dir`） |
| POST | `/v1/models/load` | `{name\|repo_id, kind?, dtype?, device?}` | 加载结果（含耗时、显存占用） |
| POST | `/v1/models/unload` | `{name}` | 卸载并释放显存 |
| POST | `/v1/models/default` | `{name}` | 切换默认服务模型 |
| GET | `/v1/loras` | — | 已上传 LoRA 列表 |
| PUT | `/v1/loras/{name}` | safetensors 原始字节 | 上传/覆盖 |
| DELETE | `/v1/loras/{name}` | — | 删除 |
| POST | `/v1/cache/clear` | — | 清空两层缓存统计 |
| GET | `/v1/tunnel` | — | 当前公网 URL、provider、健康状态 |
| POST | `/v1/server/shutdown` | 确认字段 | 远程关机（方便从本地结束 Colab 会话进程） |

### 5.3 推理端点与长任务

| 方法 | 路径 | 模式 | 说明 |
|------|------|------|------|
| POST | `/v1/encode` | 同步 | 打包体请求 → 打包体响应（cond / pooled_output 等，含张量） |
| POST | `/v1/generate` | **异步 Job** | 立即返回 `202 {"job_id"}` |
| GET | `/v1/jobs/{id}` | 轮询 | `{"state": "queued\|running\|done\|error", "text"?, "error"?}` |
| DELETE | `/v1/jobs/{id}` | — | 尽力取消 |

- Job 结果保留 30 分钟，断线/换 URL 后凭 job_id 仍可取回；同一连接层不做自动重试（继承旧版"generate 不重复执行"原则）。
- encode 在隧道下超过 ~90s 未完成时返回 `503 retryable`（客户端可重试，幂等由 embedding 缓存兜底）。

### 5.4 数据面打包格式 `application/x-rcp-v3`

沿用旧版已验证的二段式，套进 HTTP body：

```
[8 字节大端长度 N][N 字节 JSON 元数据][二进制张量 blob]
```

- JSON 元数据：`text`、`kwargs`（可含 `__tensor__` 引用）、`gen_kwargs`、`lora_stack`、`tensors`（manifest：dtype/orig_dtype/shape/offset/size）。
- **浮点张量传输一律转 fp16**（已确认：纯远端推理，固定 fp16，不提供 fp32/auto 选项）；`orig_dtype` 在客户端按需还原（bf16 计算结果经 fp16 传输后还原为原 dtype）。
- 响应同构：`struct`（嵌套结构 + 张量引用）+ `tensors` manifest + blob。
- dtype 白名单、头部/blob 体积上限（8MB / 512MB）与旧版一致。

### 5.5 认证与安全

- token：`--token` 显式指定，或隧道模式下自动生成（secrets，32 hex）随 URL 打印。
- TLS 由隧道提供；`direct` 模式下明文 HTTP，README 明示仅限可信局域网。
- 上传接口校验 safetensors 魔数；所有文件写入限制在运行时自身 `models/` 目录内（路径穿越校验）。
- 鉴权比较使用常数时间比较。

---

## 6. 本地加载侧（ComfyUI 自定义节点）

新包名（默认，待确认 Q3）：`ComfyUI-RemoteCLIPColab`，节点目录仍是 `Remote CLIP`。

### 6.1 节点清单

| 节点 | 输入要点 | 行为 |
|------|----------|------|
| `LoadRemoteCLIPColab`（Load Remote CLIP Colab） | `base_url`（如 `https://xxx.trycloudflare.com`）、`auth_token` | 连接时调 `/v1/status` 校验协议与能力 → 返回 `RemoteCLIPProxy`（HTTP 版）；传输固定 fp16，无精度选项 |
| `LoraLoaderCLIPOnly`（与旧版同名同语义） | `CLIP`、`lora_name`、`strength_clip` | 远程代理 → 转发名称与强度（远程在 Colab 端查找/应用）；本地 CLIP → 原地应用（兼容混用） |
| `RemoteCLIPController`（可选，Q4） | `base_url`、`auth_token`、`action`（status/load_model/unload_model/set_default/clear_cache/list_models/list_loras/shutdown）、参数字段 | 直连 REST 控制面，结果 JSON 显示在节点 UI 文本里，实现"本地侧控制远程端"的可视化 |

### 6.2 代理对象（对工作流完全兼容）

`RemoteCLIPProxy` 保持旧版鸭子类型接口不变：`tokenize / encode_from_tokens(_scheduled) / generate / decode / clone / with_lora / add_patches（明确报错并指引用 LoraLoaderCLIPOnly）`，内部改为 HTTP 调用；`tokenize` 仍在本地侧收集文本与 kwargs（保持零远程往返的轻路径），编码/生成在远程执行。LoRA 名称列表来自 `/v1/loras`（远程下拉可选，保留手填）。

### 6.3 功能保留对照（G4 验收基准）

| 旧版功能 | 新版去处 |
|----------|----------|
| encode（条件编码，含多模态张量输入） | `/v1/encode` ✅ |
| generate（生成式编码器出文本） | `/v1/generate` + Job ✅ |
| LoRA 转发 + 远程应用 | `lora_stack` 随请求 + LoRA 引擎 ✅；补丁保持 + 换栈还原（替代旧版 LRU 补丁缓存，同样避免重复补丁开销） |
| embedding 缓存 | ✅ 同参数（缓存条目固定存 CPU 副本，不占远端显存） |
| auth token | Bearer + 常数时间比较 ✅（隧道下强制） |
| transport_precision auto/fp16/fp32 | **简化**：固定 fp16 传输（用户确认），fp32/auto 选项移除 |
| 协议版本显式拒绝 | `/v1/status` proto 协商 ✅ |
| 自动重连重试（encode）/ generate 不重试 | HTTP 无状态重试 + Job ✅ |
| `SendRemoteCLIP`（ComfyUI 内 Sender 节点） | **移除**，由独立运行时取代（Q5） |

本地侧实现仅用 Python 标准库（`urllib`/`http.client`）+ 已有依赖（torch/numpy），**不新增 requirements**。

---

## 7. 性能与限制

- 隧道路径：Cloudflare 免费层 ~100MB/请求、~100s 首字节超时（已用 Job 化与体积指引规避）；encode 典型 T4 耗时（clip-l <1s，t5-xxl 长 prompt 5–20s）在限内。
- 直连路径：无上述限制，body 上限 512MB。
- 带宽：fp16 传输默认（远程地址时），同旧版。
- 首次模型下载/加载时间不计入请求超时（加载是显式控制面操作）。

---

## 8. 交付物与目录结构

```
custom_nodes/ComfyUI-RemoteCLIPColab/
├── __init__.py          # 本地侧节点注册（Loader / LoraLoader / Controller）
├── client.py            # REST 客户端 + v3 打包/解包 + 代理对象
├── REQUIREMENTS.md      # 本文档
├── README.md            # 双端使用说明 + Colab cell 模板 + 显存指引
└── colab/               # 远程运行时（整目录独立，可单独拷贝/克隆到 Colab）
    ├── worker.py        # CLI 入口（参数：model/tunnel/token/host/port）
    ├── server.py        # FastAPI 应用与全部路由
    ├── engines.py       # clip / clip_composite / t5 / causal_lm 引擎 + LoRA 注入 + 缓存
    ├── packing.py       # v3 打包格式（与本地侧 client.py 同构）
    ├── tunnel.py        # cloudflared / ngrok / localtunnel / direct
    └── requirements.txt # Colab 侧依赖
```

---

## 9. 验收标准（端到端）

1. Colab T4 单 cell 启动 → 打印 `https://*.trycloudflare.com` 与 token；浏览器打开 `/docs` 可用。
2. 本地工作流：`LoadRemoteCLIPColab` + `CLIPTextEncode`（SD1 clip-l）→ 出图，cond 形状正确。
3. SDXL `clip_composite`：cond 拼接维度与 pooled 正确，出图正常。
4. `LoraLoaderCLIPOnly` 转发 → 远程应用成功；重复请求命中补丁缓存（status 可见命中数）。
5. `causal_lm`（gemma-3-4b-it）：`Generate Text` 节点经隧道返回文本（Job 轮询 >100s 场景用大 max_tokens 验证不被 CF 掐断）。
6. `RemoteCLIPController`：完成 status / load / set_default / clear_cache / shutdown 全链路。
7. 错误路径：无 token → 401；错 LoRA 名 → 明确 404 文案；协议不符 → 明确拒绝文案。
8. 断连恢复：杀死并重启隧道 → 换新 URL 填回节点 → 立即恢复可用（HTTP 无状态）。

---

## 10. 确认结论（2026-08-16，已定稿）

| # | 问题 | 结论 |
|---|------|------|
| Q1 | Phase 划分 | **合并**：SD3.5 三编码器、Flux 双编码器、Qwen-Image 与基础家族同步实现 |
| Q2 | 隧道默认 Cloudflare quick tunnel，ngrok/localtunnel/direct 备选 | ✅ 按建议 |
| Q3 | 包名 `ComfyUI-RemoteCLIPColab`、类别 `Remote CLIP` | ✅ 按建议 |
| Q4 | `RemoteCLIPController` 控制节点 | ✅ 需要 |
| Q5 | 不保留旧版 `SendRemoteCLIP`、不做 v2 互通，旧包独立共存 | ✅ 按建议 |
| Q6 | 远程运行时随包放 `colab/` 子目录 | ✅ 按建议 |
| Q7 | TPU v5e 支持（设备抽象 + bf16 + 形状分桶 + 预热；generate 速度列为已知限制） | ✅ 纳入 |
| Q8 | 传输精度 | **固定 fp16**（纯远端推理；移除 auto/fp32 选项） |
