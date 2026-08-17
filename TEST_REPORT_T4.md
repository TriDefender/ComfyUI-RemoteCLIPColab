# Colab T4 实测报告 — Remote CLIP Colab Worker

日期：2026-08-17 · 方式：google-colab-cli 0.6.0（WSL）申请真实 Colab T4 实例

## 环境

| 项 | 值 |
|----|-----|
| VM | Colab GPU runtime（asia-southeast1） |
| GPU | Tesla T4, 15.6 GB |
| Python / torch / transformers | 3.12.13 / 2.11.0+cu128 / 5.13.1 |
| 运行时上传 | colab/ 打包 zip（11.8 MB），setup 18 s（依赖镜像预装） |
| 权重 | comfyanonymous/flux_text_encoders：clip_l (0.25 GB) + t5xxl_fp16 (9.79 GB)，VM 内 huggingface_hub 经本机转发凭据下载（VM 出口 IP 被 HF 匿名限流） |
| worker 启动 | `--engine native --tunnel cloudflare --model flux:clip_l+t5xxl`，后台 nohup，`--lifetime 5400` |
| 公网入口 | Cloudflare quick tunnel（worker 自动拉起 cloudflared，URL 即时打印/可查询） |

## 结果：本地 Windows → 公网隧道 → T4 全链路 9/9 PASS

| # | 测试 | 耗时 | 结果 |
|---|------|------|------|
| 1 | 协议协商（/v1/status proto=3） | 1.1 s | PASS |
| 2 | 状态（gpu=Tesla T4, default=flux） | 2.0 s | PASS |
| 3 | flux 组合 encode（native 后端，clip_l+t5xxl） | 3.4 s | PASS — cond (1,256,4096) / pooled (1,768) |
| 4 | embedding 缓存命中 | 2.2 s | PASS — hits=1 |
| 5 | emphasis 权重 `(fennec:1.4)` | 3.0 s | PASS |
| 6 | 热切换 clip_l（native sources 加载 1.0 s） | 3.3 s | PASS — cond (1,77,768) |
| 7 | 切回 flux 默认 | 3.6 s | PASS |
| 8 | 错误路径（LoRA 404 / 模型路径 404） | 1.5 s | PASS |
| 9 | 清空缓存 | 0.5 s | PASS |

worker 日志（存档 rcp_t4_worker.log）确认全部请求经 Cloudflare 边缘（41.79.x.x）到达 VM。
单次 encode 公网往返 ~3 s（含隧道 RTT），fp16 传输。

## 过程中发现并处理的问题

1. **google-colab-cli 0.6.0 + jupyter-kernel-client 1.0.1 不兼容**（`KernelClient`/`JupyterSubprotocol` 属性缺失）→ 降级 jupyter-kernel-client==0.15.0 解决（CLI 侧问题，与本项目无关）。
2. Colab VM 的 HF 匿名访问被 401（共享 IP 限流）且 `cdn-lfs` DNS 解析失败 → 走本机 HF 凭据（token 文件上传至 VM 标准 location）+ hub API 下载成功；注意 Colab UI secrets（userdata）在 headless CLI 内核不可用。
3. CLI `exec` 对长时间无输出的单元有内核超时 → 大文件下载与 worker 常驻均改为 VM 内 nohup 后台 + 轮询标志，属测试编排适配，无需改项目代码。
4. Comfy-Org/flux_text_encoders 对该账号 gated → 改用公开镜像 comfyanonymous/flux_text_encoders（同一组文件）。

## 结论

- 在真实 Colab T4 上：**native 后端（vendored ComfyUI 栈）+ Cloudflare quick tunnel + RESTful 控制 + fp16 传输**全链路按设计工作。
- RemoteCLIP_Colab.ipynb 的 cell 3（启动+隧道等待）与 cell 5（控制面板 API 用法）所依赖的服务端行为均在真实环境验证。
- 无需代码修复；未覆盖项：TPU 运行时（hf 后端）与 causal_lm 生成（VM 上无生成式权重），二者已在本地验证。
