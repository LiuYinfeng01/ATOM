# Decode 的 AITER kernel 与 mHC 替换预算

[返回全模型图册](../../README.md) · [40 层 shape / dtype 讲解](README.md)

跨引擎对照另见：[gfx942 ATOM vs gfx950 vLLM 的实际 kernel、调用数与前后顺序](mi308x-mi355x-traces.md)。

**结论（2026-11-26）：MI308X 上保留 Triton。** 本机 rank 0 trace 中，准备替换的 80 次 delayed mHC + RMSNorm 合计 **1.480769 ms/decode 步**，不是能直接减掉的延迟。在同服务镜像、同型号单卡、80 套真实权重的 A/B 中，最快实测的 AITER generic HIP 比 Triton **慢 1.257–1.268 ms/80 次**，尽管发射数从 160 变为 80。原 TP4 卡组未通过空闲检查，**没有测出 HIP 服务的 E2E**；第 7 节单列条件估算和零成本上限。

### 1. Decode：实际调用了哪些 AITER kernel？

下表保留提供的 **rank 0、24 个 decode 步**统计。**ms/步是该行所有调用的 GPU duration 之和再除以步数，不是单次调用时长。** 名称相近的模板已合并；`入口 → 符号` 区分 Python/API 入口与设备上实际发射的 kernel。AITER 不是一种实现语言，表内既有 Triton 也有其他后端。

| 模块 | 实际 GPU kernel / AITER 入口 | 次/步 | ms/步 |
|---|---|---:|---:|
| mHC：带 post 的主计算 | `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel_*_HAS_POST_1` | 77 | 0.661 |
| mHC：不带 post 的主计算 | `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel_*_HAS_POST_0` | 3 | 0.019 |
| mHC：归并 | `_mhc_fused_post_pre_delayed_rmsnorm_reduce_kernel_*` | 80 | 0.801 |
| mHC：独立 post | `aiter::mhc_post_kernel` | 3 | 0.013 |
| MoE：专家 gate/up | `fused_moe` → `moe_2stage_gateup_batch1_0` | 40 | 0.741 |
| MoE：专家 down | `fused_moe` → `moe_2stage_down_batch1_0` | 40 | 0.413 |
| MoE：路由选专家 | `topk_gating_fwd` → `aiter::topk_gating_kernel_opt` | 40 | 0.312 |
| TP 通信：all-reduce | `lamportAllReduce` → `aiter::lamportAllReduceWaveSplit` | 81 | 0.708 |
| TP 通信：all-gather | `aiter::allgather_lastdim` | 3 | 0.117 |
| 激活量化 | `dynamic_per_token_scaled_quant_kernel`，两个模板 | 82 | 0.389 |
| 分组量化 | `dynamic_per_group_scaled_quant_kernel` | 8 | 0.040 |
| Q/K 归一化与量化 | `fused_qk_rmsnorm_group_quant_kernel` | 40 | 0.190 |
| RMSNorm | `add_rmsnorm_quant_kernel`，两个模板 | 5 | 0.025 |
| 缓存 RoPE | `kn_entry_1c_sbhd_cached_indirect_inplace` | 12 | 0.115 |
| 小尺寸 BF16 矩阵向量乘 | `wv_splitk_small_fp16_bf16_kernel` | 12 | 0.058 |
| Indexer：FP8 logits 打分 | `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | 8 | 0.046 |
| Indexer：长行选 Top-512 | `aiter::ob::radix_topk_one_block_kernel` | 4 | 0.266 |
| Indexer：紧凑行选 Top-512 | `topk_per_row_decode_1wg_k512_wave64_wv0` | 4 | 0.101 |
| Indexer：选 2,048 个候选块 | `topk_per_row_decode_1wg_k2048_wave64_wv0` | 1 | 0.022 |
| **可明确归属 AITER 的合计（提供值）** | | **543** | **5.038** |

逐行按显示精度相加为 **5.037 ms**，比提供的总计少 **0.001 ms**，在逐行取三位小数的舍入范围内。不能把 5.038 当作整步 TPOT，也不能把四卡的数字相加。

**与原始 trace 对账**：模型 `DeepSeek-V4.1-Flash-1017-FP4PTPC`，ATOM `5f6bdb75` / AITER `ad0d5e20`，4×MI308X，TP4、BS1、无推测解码。分析的是 2026-11-24 16:25 UTC 的 **100000 输入、24 输出**诊断请求，命中 89984；原文件时间戳为 `20261124_162528_990`。同一 trace 另有 8192、1952 行的两个 prefill 标记，已排除。取全部 24 个 `decode[bs=1 …]` 标记；只认同线程、位于标记内且 correlation 唯一的 HIP launch，再归属其 GPU events，不按设备时间戳硬截 CPU 窗。每一步的 19 项调用次数都逐项相等；**未舍入的 AITER 行和 5.037664 ms/步**，与表中 5.038 一致。不是之前图册的 32 步中段平均。

### 2. 剩下的 424 次是什么？

| rank 0 decode 的其余调用 | 次/步 | GPU ms/步 | 归属依据与边界 |
|---|---:|---:|---|
| `kernel_gemm_0` | 130 | 0.908 | 通用符号。在 8192 行的 eager prefill，65 次的直接 CPU 父算子都是 `aiter::gemm_a8w8_bpreshuffle`。decode 是 graph replay，无法据此为这 130 次逐次指定投影或 CPU 父算子。 |
| `Cijk_*` GEMM（5 个不同符号） | 85 | 1.045 | ROCm BLAS / Tensile 风格的设备符号；在 eager 路径见 `aten::mm` / `aten::bmm`。不按 AITER 自有 kernel 计。 |
| ATOM 自有 Triton | 203 | 1.327 | 其中 sparse attention 的 split、combine 和普通双源 kernel 共 **78 次、0.706 ms**；`_rope_window_kernel`、`_inverse_rope_gptj_kernel` 各 40 次。 |
| PyTorch/ATen 与 ROCm runtime | 6 | 0.031 | 张量操作与 `__amd_rocclr_copyBuffer`。 |
| **其余小计** | **424** | **3.311** | 按未舍入值计算。 |
| **加上第 1 节后的总计** | **967** | **8.349** | profiler 内可归属的 GPU 事件时长之和，不是客户端 TPOT。 |

同配置、不开 profiler 的 **7.490 ms 是每个后续输出 token 的中位 TPOT**，整条 100K→1500 请求的中位 E2E 是 **11.886 s**。不要把 `8.349 − 7.490` 解释为 profiling 开销，也不要把跨 stream 的 duration 相加当成一条关键路径。

### 3. Prefill：同名算子，耗时分布不同

同一份 trace 还有两个 prefill forward。每格是 **rank 0 的调用次数 / 该 chunk 累计 GPU ms**，各只有一个 forward，不是每 token 的均值。

| 模块 | 8192 行 chunk | 尾块 1952 有效行（缓冲 2048） |
|---|---:|---:|
| FP8 paged MQA logits | 3 / 36.415 | 8 / 10.232 |
| Indexer Top-K 选择 | 3 / 11.789 | 9 / 3.379 |
| mHC main + reduce + 独立 post | 84 / 51.084 | 163 / 14.411 |
| MoE gate/up + down | 40 / 67.471 | 80 / 34.228 |
| ATOM sparse attention | 20 / 134.035 | 41 / 37.543 |

为什么不能笼统说“FP8 logits 不贵”？decode 只有一行查询，8 次打分合计 **0.046 ms/步**；首个 prefill chunk 成批处理查询，3 次就花 **36.415 ms/chunk**。这是两种不同规模的工作。优化稳态 TPOT 看 decode 的每步成本；优化长输入 TTFT 要看 prefill 的分块工作及调度，而不是把 36.415 ms 乘到每个输出 token 上。

### 4. 瓶颈在 mHC 的哪一半？

| Rank 0，每 decode 步 | 调用次数 | GPU ms/步 | 单次均值 |
|---|---:|---:|---:|
| Triton main，带上个子层的 post | 77 | 0.660610 | 8.58 µs |
| Triton main，不带 post | 3 | 0.018735 | 6.25 µs |
| Triton reduce，合并 split-K、算门并完成归一化 | 80 | 0.801423 | 10.02 µs |
| **拟替换的 main + reduce** | **160 发 / 80 个子层边界** | **1.480769** | **18.51 µs/边界** |
| 独立 `aiter::mhc_post_kernel`，保留 | 3 | 0.012736 | 4.25 µs |

40 层各有 attention 和 FFN 两个 mHC 边界。入口和两个 Engram 插入点使 **3 次 main 不带 post**；其余 77 次在 main 中消费前一个子层的输出。`reduce` 已占拟替换时长的 **54.1%**。即使将 reduce 单独假想成零，也只省 **0.801 ms/步**，不能把整个 1.481 ms 都记给它。独立 HIP post、最终 collapse、MoE 和 TP 通信都不在替换边界里。

分母也值得记：每步有 **967** 个可归属的 GPU kernel 事件，duration 和为 **8.349122 ms**。mHC 两部分占其中 **17.7%**，占表中明确归属 AITER 那部分的 **29.4%**；不是“用户 29.4% 的等待时间”。每步 Triton 核心在 rank 0 的范围为 **1.477763–1.483685 ms**。四个 rank 单独复算的均值为 **1.480769 / 1.481370 / 1.485467 / 1.495254 ms**，不相加。其余可见大项还有专家 GateUp + Down **1.154 ms**、通用符号 `kernel_gemm_0` **0.908 ms**（不能仅凭名字断言属于唯一投影）、Lamport all-reduce **0.708 ms**。所以 mHC 值得优化，但不是整步中唯一的等待源。

### 5. 单测到底替换了什么？

实跑 ATOM 使用 `ATOM_DSV41_MHC_BACKEND=auto`。在原镜像、真实 MI308X 上检查了两条路径：`auto` 不给 `hc_fn` 打 HIP 包，回落 Triton；强制 `hip` 时，ATOM 的 `pack_mhc_fn` 抛出只支持 `gfx950/gfx1250` 的错误。**这不等于 AITER 没有 gfx942 HIP kernel。** 直接调用原镜像的 `aiter.ops.mega_mhc.mega_mhc`，在 BF16 输出、提前 pack、显式 workspace 下可运行 **generic HIP**；gfx950 的 tuned stream kernels 不是这个候选。不改服务开关、不拿另一种不带 delayed pre/RMSNorm 的 `mhc_pre` 冒充。

下表的 shape/dtype 是在 **M=1、H=5120、4 流**的 GPU 单测中从参数与返回张量实际记录的；trace 独立确认了 `H_5120/BLOCK_M_16/TILE_K_32/NUM_KSPLIT_160`、20 轮 Sinkhorn 以及两种 post 分支。老 trace 没有逐元素 activation dump。权重读取同一转换 checkpoint；激活与传入门为固定 seed 构造，**不是**真实请求的隐藏值。

| 张量 | Shape | Dtype / 契约 |
|---|---|---|
| 四流 residual；传入 pre | `[1,4,5120]`；`[1,4]` | BF16；FP32。collapse 使用**上一子层传来的** pre，不是当前预测的 pre。 |
| 每个边界的 `fn` / `scale` / `base` | `[24,20480]` / `[3]` / `[24]` | 均 FP32。HIP 的 hi/lo BF16 预打包以 INT32 `[160,4,3,4,16,4]` 容器传入。一次性权重 pack 不按步计费。 |
| 可选 `sublayer_out` / `post_layer_mix` / `comb_res_mix` | `[1,5120]` / `[1,4]` / `[1,4,4]` | BF16 / FP32 / FP32；no-post 时三者均不传。 |
| 子层 RMSNorm 权重 | `[5120]` | BF16；norm 和 RMS 的 epsilon `1e-20`，mHC epsilon `1e-6`，post 系数 2，Sinkhorn 20 轮；输出 BF16，不额外做 MXFP8 输出融合。 |
| 五个输出：新残差、post、comb、normed、下一 pre | `[1,4,5120]` / `[1,4,1]` / `[1,4,4]` / `[1,5120]` / `[1,4]` | BF16 / FP32 / FP32 / BF16 / FP32。no-post 保留输入残差。 |

### 6. MI308X 单卡实测：一发 HIP 是否胜过两发 Triton？

**方法。** 物理 GPU4，MI308X / gfx942 / 80 CU；容器只映射这一张卡，两次独立进程均重新锁卡并检查显存、空闲和 KFD 所有权。使用与原服务相同的镜像（ID 前缀 `cfd987217fc4`），Torch `2.10.0+rocm7.2.4.git3d3aa833`。两种后端使用同一 stream、同一实际权重、同一输入。单步 proxy 按真实层序执行 **80 组不同参数：77 次 post、3 次 no-post**。仅 FP32 `fn` 的轮转工作集就有 **150 MiB**。这里测试的是 80 个 mHC 调用，不运行中间的 attention/MoE，不宣称重现了整模型缓存或真实数据依赖。

预热、JIT、pack、workspace 分配及 graph capture 不计时；在 GPU 上用 HIP events 包住 graph replay，保存原始配对样本。每轮先以随机顺序筛 **全部 9 个合法 split_k：1、2、4、5、8、10、16、20、32**（每个 5 组、每组 6 次整图 replay）；默认是 20，两轮筛选均为 10 最快。随后用独立计时样本 ABBA/BAAB，9 组，每块 8 次 replay。第二个进程显式保留第一轮选定的 10 复测。表中 A、B 是各自的 block 中位数；差值是配对差中位数，所以不必恰等于两个已舍入中位数相减。

| 80 个边界，单卡 graph（ms/80 次） | Triton | generic HIP | 配对增量：HIP − Triton（正值是变慢） |
|---|---:|---:|---:|
| 试验 1，默认 split=20 | 1.285720 | 2.936183 | **+1.604958** |
| 试验 2，默认 split=20 | 1.283697 | 2.849098 | **+1.564111** |
| 试验 1，最快 split=10 | 1.285627 | 2.556829 | **+1.267914** |
| 试验 2，复测 split=10 | 1.283522 | 2.540962 | **+1.257142** |

单层热缓存也同方向。固定第 0 层相同权重重复 32 次，第二轮 split=10 的**每次**带 post 为 **14.63 → 30.70 µs**，不带 post 为 **13.10 → 23.53 µs**（Triton → HIP）。不要拿这组热缓存时间乘 80 替代上表。profiler 实际观察到两种 post 模式各自都是 Triton `main+reduce` 两发，对比一发 `mega_mhc_kernel<5120,13,…>`。**少发一次是真的，完成同一工作更慢也是真的。** split 改变后时间明显改变，说明不能只按 launch 个数估收益；并未用硬件计数器证明唯一根因是带宽、占用或同步。

**数值。** 同 checkpoint 40 层 × 两种子层 × 两套 seed，共 **160 组**。独立 FP64 参考显式保留模型的 BF16 写回/舍入；逐项检查全部五个输出，另将 HIP 直接与 Triton 比。BF16 residual/norm 用 `atol=0.02, rtol=0.01`；FP32 gates 用 `atol=5e-4, rtol=1e-3`。默认与最快 split 在两轮均无非有限值、无元素超阈；第二轮还核对连续 graph replay 后的 80 组输出。split=10 对参考：新残差最大绝对差 **0.015625**、normed **0.00390625**、三个 gate 输出均不超过 **4.92e-5**。不是逐 bit 等同，也不是整模型精度验收。

### 7. 理论上能省多少？整条请求呢？

使用**与 24 步 trace 同轮的另一趟、不开 profiler 的 C1**：TP4，100000→1500，并发 1，4 条有效 measured 请求，均命中 89984。中位 TPOT **7.489738 ms**、TTFT **627.613 ms**、请求 E2E **11.886007 s**。不是 24 输出的诊断时长，也不是其他轮次的 7.509 ms。TPOT 和 E2E 的中位数分别来自各请求，不能用两组分别取出的中位 TTFT/TPOT 强拼一个新基线。

采用一个可证伪的**只改 decode 的敏感性情景**：设每个生成间隔真正省下 `d` 毫秒，TTFT 不变、无排队变化、四卡关键路径同幅变化。则 `TPOT′ = 7.489738 − d`，`E2E′ = 11.886007 − 1499 × d / 1000` 秒。1500 个输出只有 **1499** 个首 token 之后的间隔。对 HIP 采用**同一单测环境的配对差**外推，不将绝对 HIP 单测时长直接减掉 profiled 服务时长。

| 情景 | 假设省下 d（ms/步） | TPOT（ms） | 100K→1500 请求 E2E（s） | 证据等级 |
|---|---:|---:|---:|---|
| 现有 ATOM / Triton | 0 | **7.490** | **11.886** | 既有无 profiler 服务实测 |
| 替换为实测最快的 generic HIP（split=10） | **−1.268 至 −1.257** | **8.747–8.758** | **13.770–13.787** | 单算子配对差全额传导的情景，非服务实测 |
| 替换为默认 generic HIP（split=20） | **−1.605 至 −1.564** | **9.054–9.095** | **14.231–14.292** | 同上 |
| 假想未来实现：服务中 mHC 工作时间减半 | +0.740384 | 6.749 | 10.776 | 理想串行敏感性，不是当前 HIP 性能 |
| 假想 main + reduce 完全免费 | **+1.480769** | **6.009** | **9.666** | 只消除这段 GPU duration、完全在关键路径时的极乐观上限，不可当可达值 |

**如何决策。** 不给 `gfx942` 简单放开 stock HIP 开关。若目标是只靠这段省 **0.5 ms/token**，服务里的 80 个边界平均应降到 **12.26 µs/次以下**（现为 18.51）；想让整步 TPOT 降 10%，理想门槛约 **9.15 µs/次**。这比“能否合成一发”更适合作为新移植的验收指标。应先对 split-K 与占比 54.1% 的 reduce 分开测，再让正确且过门槛的候选跑同卡组 fresh-server A/B。此次原 GPU0–3 卡组有其他 GPU 作业、未通过准入；没有抢卡、没有对外发布服务补丁，也**没有 HIP 整模型 TPOT、E2E 或质量实测**。若连 prefill 也换，TTFT 必须另测，本表不适用。

### 8. 复核与复跑

| 资料 | 内容 |
|---|---|
| [rank 0 逐步复算结果](mhc-study/rank0-summary.json) · [标准库 trace 分析脚本](mhc-study/audit_trace.py) | 24 步、各 kernel 完整符号/调用数、每步 duration 与原压缩 trace SHA-256。公开的是派生统计，不是原始 trace。 |
| [单算子 A/B 及数值参考](mhc-study/bench_mhc.py) · [试验 1](mhc-study/op-round-1.json) · [试验 2](mhc-study/op-round-2.json) | 实测输入/输出 shape、全部 split 筛选、ABBA 配对样本、实际符号和数值误差。试验 2 新增 graph 重放数值验收，试验 1 没有这一额外检查。 |
| [ATOM 后端门限](https://github.com/ROCm/ATOM/blob/5f6bdb75c19a0e4bc57512ef425973d53f0f54f4/atom/model_ops/deepseek_v41/mhc_pre_delayed.py) · [AITER HIP 接口](https://github.com/ROCm/aiter/blob/ad0d5e208eb2d49ab658c3c161aa5c3502dfc2a1/aiter/ops/mega_mhc.py) · [HIP 实现](https://github.com/ROCm/aiter/blob/ad0d5e208eb2d49ab658c3c161aa5c3502dfc2a1/csrc/kernels/mega_mhc.cu) | 固定版本。已核对 HIP 接口和设备源码的 SHA-256 与实测镜像内文件一致。 |

拥有原始 trace 时，从本页目录执行 `python mhc-study/audit_trace.py RANK0_TRACE.json.gz rank0-local.json`。在有同版 ROCm PyTorch、ATOM、AITER 和 checkpoint 的环境里，先保留一张**独占** MI308X，仅使该卡可见，然后运行 `python mhc-study/bench_mhc.py "$MODEL_DIR" op-local.json --include-split 10`。`--smoke` 只测第 0 层两套参数，不能拿它代表全模型。两项都不需要复制或上传 checkpoint、原始激活或 prompt。