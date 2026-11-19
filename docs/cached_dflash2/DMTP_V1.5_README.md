# D-MTP v1.5：离线数据准备、训练与交付

本文说明如何使用本分支的 cached DFlash2 trainer 训练 D-MTP v1.5。相较 v1.0，本版本在 draft 输入中保留两个 anchor：pending token `A` 和 MTP 提出的首候选 `T1`；当使用 `fusion` 配方时，MTP 预测 `T1` 时的 hidden 通过零初始化残差注入 `T1` embedding。DFlash 并行预测后六个 token `T2…T7`。

```text
目标模型已输出 A，尚未处理 A
原生 MTP(A, target hidden 至 A 之前) -> T1、h_mtp
DFlash 输入 [A, T1 + Δ, MASK, MASK, MASK, MASK, MASK, MASK]
DFlash 预测 T2…T7；target 验证 T1…T7
```

实现入口：[`src/speculators/train/cached_dflash2/dmtp.py`](../../src/speculators/train/cached_dflash2/dmtp.py)、[`data.py`](../../src/speculators/train/cached_dflash2/data.py)、[`trainer.py`](../../src/speculators/train/cached_dflash2/trainer.py)。本文中的 shell 命令使用 Linux/Bash，路径按你的训练机修改。

## 1. 先确认你手里的数据

训练器消费的是离线缓存，不会自行加载图片、跑目标模型或采集 MTP 状态。每个样本至少需要：

1. 完整的 token IDs，且与采集 hidden 时的 tokenizer、模板、图像处理和位置完全一致。
2. 目标模型在草稿配置指定层的 hidden states，用作 DFlash context features。
3. 目标模型 LM head 的输入 hidden，用于当前训练器的缓存教师评估；v1.5 正式训练 loss 的标签来自真实 token IDs。
4. 原生 MTP 的预测时 hidden 和对应候选 token；必须能按 token 位置与 IDs 对齐。
5. 初始 dense DFlash2 checkpoint，以及与 target 匹配的共享 embedding 和 LM head。

你已有的第 5、15、25、35、45 层 hidden states 可作为 DFlash 的五路 context features，前提是它们确实对应初始草稿 `config.json` 的 `dflash_config.target_layer_ids`，且按该顺序拼接。你说的最后一层输出只有在它是**目标 LM head 实际接收的 hidden**时，才能作为 `teacher_hidden` 供缓存评估使用；v1.5 训练标签仍取自 token IDs。MTP 数据还必须含有预测 `T1` 时的单流 hidden；只有 MTP 输入和生成的 token 不够做 v1.5 的 fusion 训练。

先从 checkpoint 核实：

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("/path/to/dflash2-initial/config.json").read_text())
print("target_layer_ids:", config["dflash_config"]["target_layer_ids"])
print("hidden_size:", config["hidden_size"])
print("vocab_size:", config["vocab_size"])
PY
```

层编号必须精确匹配；如果这里不是 `[5, 15, 25, 35, 45]`，不要仅靠重排文件名或改 manifest 来假定兼容。先确认 layer ID 是从 0 还是从 1 计数，以及采集代码拿的是 block 输出还是 embedding 输出。

## 2. v1.5 的位置与时间约定

令 `a` 为已发给用户但 target 尚未处理的 pending token `A = ids[a]`。目标上下文只包括位置 `< a`。原生 MTP 使用 `A` 和 target 在 `a-1` 处的状态，产生用于预测 `T1 = ids[a+1]` 的 hidden `h_mtp[a]` 及候选 token。

训练时使用真实序列的 `ids[a+1]` 作为 T1，避免把离线缓存的随机采样候选当作训练真值。验证和推理时，T1 使用原生 MTP 实际产生的 token。两种场景融合的 MTP hidden 都必须是**预测 T1 时的状态**，不能使用 MTP 消费 T1 后的状态，也不能用 target 最后一层 hidden 替代。

Fusion 计算为：

```text
Δ = W([RMSNorm(E(T1)); RMSNorm(stopgrad(h_mtp[a]))])
E'(T1) = E(T1) + Δ
W 初始化为 0
```

零初始化保证训练刚开始时 `E'(T1) = E(T1)`。残差只加在 T1 槽位；A 和六个 MASK 槽位不直接加 MTP 残差。八个输入槽对应 RoPE 位置 `a…a+7`。模型只取最后六个槽的输出作为 T2…T7 logits，监督目标 token 位置为 `a+2…a+7`。

## 3. 离线缓存格式

推荐每条序列一个 target cache、一个 MTP sidecar，由 manifest 配对：

```text
/data/dmtp-v15/
  manifest.json
  capture-provenance.json
  shared.pt
  target/sample-0001.pt
  mtp/sample-0001.pt
```

Manifest 是 UTF-8 JSON 数组，示例：

```json
[
  {
    "id": "sample-0001",
    "split": "train",
    "image_sha256": "<原始样本身份摘要>",
    "tokens": 1024,
    "prompt_length": 600,
    "path": "target/sample-0001.pt",
    "mtp_path": "mtp/sample-0001.pt"
  },
  {
    "id": "sample-0002",
    "split": "val",
    "image_sha256": "<另一条样本的身份摘要>",
    "tokens": 900,
    "prompt_length": 500,
    "path": "target/sample-0002.pt",
    "mtp_path": "mtp/sample-0002.pt"
  }
]
```

路径可相对 manifest，也可为绝对路径。train 和 val 必须都非空，且 `image_sha256` 不得跨 split 重复。`tokens` 是序列长度 L；`prompt_length` 是 token 数，必须和 cache 一致。低于当前最短回答门限的 train/val 样本会被排除，具体数量写入 provenance。

### Target cache：`path` 指向的 `.pt`

用 `torch.save` 保存普通 dict；读取端使用 `torch.load(weights_only=True)`。必须包含：

| 字段 | 形状和类型 | 语义 |
|---|---|---|
| `ids` | `[L]`，int32/int64 | 完整多模态 token 序列，和 hidden 使用同一位置索引 |
| `features` | `[L, 5H]`，BF16 | 五层 collapsed hidden 按 `target_layer_ids` 顺序拼接；此处 H 是 draft hidden size |
| `teacher_hidden` | `[L, H]`，BF16 | target LM head 输入；位置 i 对应预测 `ids[i+1]` 的分布，用于缓存教师评估和旧蒸馏配方，不是 v1.5 token CE 标签 |
| `prompt_length` | Python int | 与 manifest 完全相同 |
| `image_sha256` | str | 与 manifest 完全相同 |

若模型实际的 `target_layer_ids` 数量不是 5，`features` 的第二维必须改为 `N×H`，并确认初始草稿 fc 层与该 N 相符。`teacher_hidden` 不能是 logits、右移后的张量或未经确认的“最后一层输出”。

### MTP sidecar：`mtp_path` 指向的 `.pt`

必须含有与 target cache 相同的 `ids` 和 `image_sha256`，以及：

| 字段 | 形状和类型 | 语义 |
|---|---|---|
| `mtp_sample_hidden` | `[L, H]`，BF16 | 原生 MTP 单流 LM head 输入，即预测 `ids[a+1]` 时使用的 hidden |
| `mtp_token` | `[L]`，int64 | 对应位置 MTP hidden 实际产生的首候选 token；用于验证/评估 |
| `mtp_valid` | `[L]`，bool | 标记采集有效位置；被选作 anchor 的位置必须为 true |
| `history` | str | 必须为 `native_causal_target_aligned` |
| `alignment` | str | 必须为 `pending_A_at_a__target_HC_at_a-1__predict_T1_at_a+1` |

Target cache 和 MTP sidecar 必须基于同一序列、同一 tokenizer 和相同多模态 token 展开。`capture-provenance.json` 建议保存 target/MTP 权重摘要、权重精度与量化方式、tokenizer/template/processor 版本、target layer IDs 和拼接顺序、采集代码版本、生成参数及时间对齐说明。训练器的 `--check-data` 不会替你验证这些来源声明。

### `shared.pt` 与初始 checkpoint

`shared.pt` 是由 `torch.save` 保存的 dict，包含 `embed_tokens.weight` 和 `lm_head.weight`，形状均为 `[vocab_size, hidden_size]`，来自产生这些缓存的同一个 target。初始 dense checkpoint 至少含匹配的 `config.json` 和 `model.safetensors`。Dense、token-only 和 fusion 对照应从**同一份** dense 初始化开始，不要把其中一组的训练结果当作另一组的起点。

## 4. 环境准备与数据预检

在 Linux/CUDA 训练机的仓库根目录安装依赖。训练过程不需要启动 vLLM：

```bash
uv sync --extra dev
source .venv/bin/activate
speculators train-cached-dflash2 --help
```

设置本机路径。示例使用 Bash：

```bash
export DFLASH2_INITIAL=/models/dflash2-dense
export DFLASH2_SHARED=/data/dmtp-v15/shared.pt
export DMTP_CACHE_MANIFEST=/data/dmtp-v15/manifest.json
export DMTP_RUN_ROOT=/runs/dmtp-v15
```

先用 v1.5 fusion 配置扫描缓存。该命令读取数据并验证格式，不开始训练：

```bash
speculators train-cached-dflash2 \
  --config examples/cached_dflash2/fusion-v1.5-token-ce-3e-5.yaml \
  --check-data
```

`--check-data` 会检查 manifest、缓存文件、样本身份、split、张量形状/dtype/有限值和 MTP 有效 anchor 等。它不会证明 target hidden 真来自宣称的层、MTP hidden 的时序语义真实正确，也不会验证初始权重来自同一个 target；这些必须由采集记录及离线审计确认。

## 5. 先跑独立 pilot

每个新配置先用单独的输出目录做短 pilot，不要复用正式训练输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 speculators train-cached-dflash2 \
  --config examples/cached_dflash2/fusion-v1.5-token-ce-3e-5.yaml \
  --output "$DMTP_RUN_ROOT/fusion-pilot" \
  --pilot-steps 3 --pilot-val-images 8
```

检查进程退出码、`train.jsonl` 中实际更新步数、loss 是否有限、`summary.json` 中 `selector_unchanged` 是否为 true，以及 checkpoint 是否成功导出。Pilot 只用于发现格式、内存和数值问题，不用于报告训练效果或部署。

## 6. 正式训练 v1.5 fusion

仓库的 `examples/cached_dflash2/fusion-v1.5-token-ce-3e-5.yaml` 已选择 `arm: fusion`、`fusion_init: residual-zero` 和 transcript token CE。确认文件内变量对应当前会话设置后，启动正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 speculators train-cached-dflash2 \
  --config examples/cached_dflash2/fusion-v1.5-token-ce-3e-5.yaml
```

也可以复制 YAML 到本次实验目录，显式写全字段：

```yaml
dense_checkpoint: ${DFLASH2_INITIAL}
shared: ${DFLASH2_SHARED}
manifest: ${DMTP_CACHE_MANIFEST}
output: ${DMTP_RUN_ROOT}/fusion-v1.5
arm: fusion
fusion_init: residual-zero
objective: token_ce
lr: 3e-5
epochs: 5
warmup_fraction: 0.2
batch_sequences: 32
context_token_budget: 48000
anchors: 64
eval_batch_sequences: 4
eval_anchors: 8
seed: 280
memory_limit_gib: 96
cache_target_precision: fp8
```

配置数值是实验起点，不保证适合每种显卡、上下文长度或数据集。`cache_target_precision` 必须填真实 target 权重采集精度，不能根据 cache dtype 推断。该 trainer 为单 GPU、单进程流程；不要用 torchrun 对同一 output 运行多个 worker。训练会记录命令、源码和输入摘要、配方、日志及 epoch 导出。恢复训练时使用原输出目录的 `run.yaml` 并传 `--resume`；输入、源码、配方和权重摘要需要一致：

```bash
CUDA_VISIBLE_DEVICES=0 speculators train-cached-dflash2 \
  --config "$DMTP_RUN_ROOT/fusion-v1.5/run.yaml" --resume
```

## 7. 训练目标与运行产物

Fusion 模式冻结 target、原生 MTP、共享 embedding/head 和 candidate selector，训练 DFlash 主干及 `anchor_fusion`。训练样本的 T1 是 gold token `ids[a+1]`；只对 DFlash 后六个输出位置计算 loss，标签是 transcript 中的真实 token `ids[a+2:a+8]`。v1.5 使用 DFlash 的位置衰减交叉熵：每个位置按 `exp(-slot/4)` 加权并归一化到均值为 1。不要把 `objective: kl` 或 `objective: ce` 当成同一训练目标：这两个旧配方分别是冻结教师分布 KL 和教师 argmax CE。v1.5 示例配置为 `examples/cached_dflash2/fusion-v1.5-token-ce-3e-5.yaml`。教师 hidden 仍用于缓存评估，但不会作为 v1.5 token CE 的标签。

主要产物位于配置的 `output`：

- `run.yaml`、`train_command.txt`、`provenance.json`：配置、运行来源、输入和代码摘要。
- `train.jsonl`、`validation.jsonl`、`initial.json`、`summary.json`：训练和缓存验证记录。
- `epochs/epoch-00…`：含 `config.json`、`model.safetensors` 和导出记录的 epoch checkpoint。
- `resume.pt`：恢复训练所需状态；只从可信的本轮运行目录加载。

缓存验证是固定 transcript 前缀上的离线代理评估，不等价于真实 speculative decoding 在线接受率或吞吐。对照实验要固定数据划分、初始化、训练预算、target 版本和评估设置。不要把 token-only / dense 的指标与不同数据、不同 target 或不同训练起点直接比较。

## 8. 评估与部署边界

验证时首候选 T1 来自 MTP，融合残差使用预测该 T1 的 MTP hidden；target 一次验证候选前缀 T1…T7。D-MTP 将原生 MTP 首候选能力与 DFlash 后六位并行起草组合起来。

**注意：训练 checkpoint 能导出，不代表通用上游 vLLM 或 SGLang 已支持它。** 当前训练改动采用双 anchor 8 槽输入；推理 runtime 必须同样消费 `[A,T1,MASK×6]`、只将残差加在 T1、保持 target context `<a`，并正确提交/回滚 MTP 和 draft 状态。旧版 v1 runtime 若仍按单 anchor 或不同 block 布局读取权重，不能直接宣称兼容。部署前应针对对应 runtime 实现完成加载检查、小样本逐 token 对齐、拒绝回滚和在线 speculative 验证；本 README 不声称双 anchor runtime 已完成这些验证。

## 9. 常见问题

**我有第 5、15、25、35、45 层 hidden 和最后一层输出，是否足够？** 这些可以分别用作 `features` 和 `teacher_hidden` 的来源，但必须确认层编号、拼接顺序、LM head 输入定义、target 权重与序列位置。还需有每个位置 MTP 预测时的 `mtp_sample_hidden` 和 `mtp_token`、IDs、prompt 边界以及匹配的共享 embedding/head、初始草稿权重。

**只保存 MTP 生成的 token 是否够？** 不够。Fusion 还要 MTP 预测该 token 时的 hidden；训练还要 gold 序列 token、Target context features 和教师监督 hidden。

**能用 target 最后一层 hidden 当 MTP hidden 吗？** 不能。二者来自不同计算路径、具有不同的预测时序。

**能只用真实 T1 训练，不采集 MTP token 吗？** 训练 T1 来自 gold transcript，但 MTP hidden 必须有；要做缓存验证和在线推理还需要实际 MTP token 及相应 runtime。

**这套数据能否直接拿去在线部署？** 不能仅凭训练成功判断。训练、checkpoint 导出和目标推理 runtime 是三个独立验证步骤。

## 参考文档

- [通用缓存训练契约](TRAINING.md)
- [业务数据接入和完整运行步骤](BUSINESS_DATA.md)
- [v1.0 cached DFlash2/D-MTP 记录和实验归档](README.md)
- [固定 vLLM runtime 的安装及验证说明](VLLM_INFERENCE.md)
- [D-MTP 算法说明](https://github.com/Sylvan820/D-MTP/blob/main/DMTP.zh.md)
