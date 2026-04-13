
## TinyStories 训练与采样

下面是仓库里新增脚本的常用命令说明。

### 1) 训练模型

脚本：`scripts/train_tinystories.py`  
默认使用字节级建模（`vocab_size=256`），直接读取 `TinyStoriesV2-GPT4-train.txt` 做训练。

```sh
uv run python -u scripts/train_tinystories.py \
  --data-path data/TinyStoriesV2-GPT4-train.txt \
  --max-bytes -1 \
  --steps 30000 \
  --log-every 50 \
  --save-every 2000 \
  --batch-size 24 \
  --context-length 256 \
  --d-model 512 \
  --n-heads 16 \
  --n-layers 12 \
  --d-ff 1344 \
  --lr 3e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --out-dir checkpoints/tinystories_big
```

常用参数：

- `--max-bytes`：只取训练文本前 N 个字节（快速试跑很有用）。
- `--steps`：训练步数。
- `--batch-size`：batch 大小。
- `--context-length`：每条样本的上下文长度。
- `--save-every`：每隔多少步保存一次 checkpoint。
- `--device`：可手动指定 `cpu` 或 `cuda`，默认自动检测。

checkpoint 默认保存到：`checkpoints/tinystories/`。

### 2) 采样生成文本

脚本：`scripts/sample_tinystories.py`  
从训练出的 checkpoint 读取模型并续写 prompt。

```sh
uv run python scripts/sample_tinystories.py \
  --checkpoint checkpoints/tinystories/tinystories_step2000.pt \
  --prompt "Once upon a time," \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 50
```

常用参数：

- `--checkpoint`：要加载的模型权重路径。
- `--prompt`：续写起始文本。
- `--max-new-tokens`：最多生成多少新 token。
- `--temperature`：采样温度，越大越随机。
- `--top-k`：只在概率最高的前 k 个 token 中采样（`0` 表示不限制）。

### 3) 推荐流程

1. 先用较小配置 smoke test（例如 `--steps 30 --max-bytes 500000`）。
2. 确认 loss 在下降后，再拉大 `--steps` 和 `--max-bytes` 正式训练。
3. 训练中用采样脚本观察生成质量变化，选择效果更好的 checkpoint。