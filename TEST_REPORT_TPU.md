# Colab TPU v5e 实测报告 — Remote CLIP Colab Worker (hf 后端)

日期：2026-08-17 · 方式：google-colab-cli 0.6.0（WSL）申请真实 Colab TPU v5e1 实例

## 环境

| 项 | 值 |
|----|-----|
| VM | Colab TPU runtime（tpu-v5e1-s，euw4b1） |
| 加速器 | TPU v5e（单芯，`xla:0`） |
| Python / torch / torch_xla / transformers | 3.12 / 2.9.0+cpu / **2.9.0（镜像预装）** / 5.13.1 |
| 依赖安装 | 选择性安装（**跳过 torch**，避免破坏 torch↔torch_xla 配对），14 s |
| 运行时 | colab/ zip 上传；worker 以 `--engine hf --device tpu` 启动（TPU 不走 vendored 栈，按设计强制 hf） |
| 模型 | `models.json`：flux 组合（clip_l + t5xxl_fp16，dtype=bf16），经 comfy 内置 tokenizer 离线加载 |
| 公网入口 | Cloudflare quick tunnel 自动拉起 |

## 结果：本地 Windows → 公网隧道 → TPU v5e 全链路 9/9 PASS

| # | 测试 | 耗时 | 结果 |
|---|------|------|------|
| 1 | 协议协商 | 0.8 s | PASS |
| 2 | 状态（device=xla:0, gpu type=xla） | 1.2 s | PASS |
| 3 | flux encode（**首次含 XLA 编译**） | **9.9 s** | PASS — cond (1,256,4096) / pooled (1,768) |
| 4 | embedding 缓存命中 | 2.0 s | PASS |
| 5 | 同 bucket 新 prompt（编译缓存复用） | **1.4 s** | PASS — 7× 加速，验证 TPU_BUCKETS 分桶策略 |
| 6 | emphasis 权重（新 batch 形状→再编译） | 9.6 s | PASS |
| 7 | 热切换 clip_l↔flux（clip_l 77 定长编译一次） | 15.8 s | PASS — cond (1,77,768) |
| 8 | 错误路径（LoRA 404 / 路径 404） | 0.8 s | PASS |
| 9 | 清空缓存 | 0.5 s | PASS |

worker 日志（存档 rcp_tpu_worker.log）确认请求全部经 Cloudflare 边缘到达。

## 发现并修复的 bug

**客户端 encode socket 超时 120 s 硬编码**（`client.py`）：TPU 首次 encode 含 XLA 编译可达
数十秒~分钟级，服务端 encode-timeout 默认 600 s，但客户端会先断线。已修复：新增
`ENCODE_TIMEOUT = 600` 并用于 encode 请求（本地侧修复，当轮即生效；T4/CUDA 不受影响，
因其首次编码 <5 s）。

## 环境备注（非项目缺陷）

1. TPU 镜像 torch 为 CPU build 且与 torch_xla 2.9.0 配对预装——安装 requirements 时**必须跳过 torch**（notebook cell 1 的 `-r requirements.txt` 在 TPU 上会拉 CUDA torch 破坏配对；本次实测以选择性安装规避，后续应在 notebook/requirements 文档标注）。
2. 首次 XLA 探测进程若被内核超时打断会持有 `/dev/vfio/0`，后续初始化报 `Device or resource busy`——kill 残留进程即恢复。
3. HF 匿名下载在 Colab VM 上被 IP 限流（同 T4 轮次），沿用本机凭据转发方案。

## 结论

- **TPU v5e 全链路按设计工作**：hf 后端 bf16 计算、形状分桶限制重编译、mark_step 同步、
  Cloudflare 隧道、RESTful 控制、fp16 传输全部验证通过。
- 性能特征符合预期：编码路径编译后 1.4 s/次（同 bucket），首次编译 ~10 s；generate
  （自回归）未在本轮测试（无生成式权重），维持"TPU 上 generate 慢于 GPU"的已知声明。
- 与 T4/CUDA 轮次合并：**CUDA MVP 9/9 + TPU 9/9，双后端在真实 Colab 上均验收通过。**
