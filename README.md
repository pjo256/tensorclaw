# TensorClaw

AI researcher harness built on [pi-mono](https://github.com/badlogic/pi-mono), inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Core loop

`tensorclaw` enforces metric-gated keep/discard decisions, automatic git revert, and a durable experiment ledger (`.tensorclaw/results.tsv`, `journal.md`, logs) on top of `pi`.

- Agent proposes one focused change.
- Harness runs the experiment and parses metrics (`val_bpb`, `val_loss`, etc).
- Harness keeps improvements, reverts regressions with git, and records the run.


## Requirements

- Python 3.10+
- `pi` CLI on `PATH` (`brew install pi-coding-agent`)
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- A repo with a runnable train/eval command that prints the target metric

## Install

```bash
pip install -e .
```

## Run

From your target project root:

```bash
tensorclaw
```

Interactive controls:

- `Enter`: run one iteration
- `<text>`: chat with the agent
- `/run N`: run N iterations
- `/status`: show history + best
- `/tail [N]`: tail latest run log
- `/reset`: clear local TensorClaw history

## TinyStories demo

We use the [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) on [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx). See [autoresearch](https://github.com/karpathy/autoresearch/blob/master/README.md#platform-support) for more info.

cli:

```bash
tensorclaw
```

```text
> what should we try first on TinyStories?
Start with a strong baseline run: a small GPT model on TinyStories, then use val_bpb as the anchor for follow-up tuning.
> yes run that
Great—I'll kick off a baseline TinyStories run ...
Iteration 1 | baseline 2.132501
Plan: Keep model/data pipeline fixed and run iteration 1 by only reshaping the LR schedule to stabilize late training.
Edits:
- train.py: change scheduler constants to `WARMUP_RATIO=0.05`, `WARMDOWN_RATIO=0.60`, `FINAL_LR_FRAC=0.05`; leave all other hyperparameters/architecture unchanged.
- prepare.py: no change (data/tokenizer flow stays identical for clean comparability).
- Impact: This should reduce over-aggressive end-of-run LR decay/instability and has a reasonable chance to improve `val_bpb` below 2.132501.
- Risk: Longer warmdown plus nonzero final LR may underfit slightly if current run already benefits from near-zero terminal LR.
Approve this plan? [Y/n]: y
step 00050 (96.2%) | loss: 4.947064 | lrm: 0.11 | dt: 4722ms | tok/sec: 13,880 | epoch: 1 | remaining: 7s
step 00051 (97.8%) | loss: 4.927629 | lrm: 0.08 | dt: 4733ms | tok/sec: 13,845 | epoch: 1 | remaining: 2s
step 00052 (99.4%) | loss: 4.911217 | lrm: 0.06 | dt: 4689ms | tok/sec: 13,977 | epoch: 1 | remaining: 0s
Training completed in 306.6s
Final eval completed in 34.2s
val_bpb:          1.664885
Result: keep | metric 1.664885 | best 1.664885
```

Artifacts:

- `.tensorclaw/results.tsv`
- `.tensorclaw/journal.md`
- `.tensorclaw/logs/`

## License

MIT
