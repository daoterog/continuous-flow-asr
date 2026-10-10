# Run profiles

Settings that depend on *where* a run executes or *what kind* of run it is, kept in one
place instead of scattered across job scripts, sweep `COMMON` blocks and `base.yaml`.
Hyperparameters — LRs, the schedule, model sizes — are not here; they stay in the variant
configs and the sweep arms.

    python train.py --config configs/audio_cond.yaml --where snellius --run sweep

**Merge order:** `base.yaml` → `--where` → `--run` → the variant (`audio_cond.yaml`, …) →
`--size` → `--mu` → split flags → dotlist overrides. The profiles sit *under* the variant:
they replace defaults, and a variant that pins one of these keys on purpose keeps it
(`flow_mapping.yaml` keeps its 50 epochs and `compile: false`). An explicit override still
beats everything.

**Choosing:**
- `--where` is detected when omitted: `snellius` inside a SLURM job or on a Snellius node,
  else `local`.
- `--run` is omitted for ad-hoc local work (base.yaml and the variant decide). On Snellius
  `jobs/batch/train.sh` passes `--run full` unless a sweep's tier says `sweep`.
- Both land in the config as `profile.where` / `profile.run`, so every wandb run says which
  profile it ran under.

| `where/` | local | snellius | why |
|---|---|---|---|
| `data.env` | local | snellius | data root |
| `data.num_workers` | 4 | 16 | 6-core laptop vs 18-CPU allocation (a packed job splits them) |
| `data.eval_batch_size` | null | 512 | 6 GB vs 40 GB card |
| `schedule.probe_n_utts` | 2048 | 8192 | probe cost vs precision |
| `train.compile` | false | true | 129.9 → 98.7 ms/step on the A100; warmup wasted on short local runs |
| `train.precision` | null | bf16-mixed | 98.7 → 70.0 ms/step; compare bf16 runs only with bf16 runs |
| `metrics.wer_bins.train` | 10 | 0 | 68.6 vs 0.5 ms of CPU per step; ~15 s of a ~77 s epoch |

| `run/` | sweep | full | why |
|---|---|---|---|
| `train.num_epochs` | 100 | 300 | screen vs final (`TIER` in `jobs/common/sweep.sh`) |
| `sampling.every_n_epochs` | 10 | 5 | a screen is read at its endpoint |
| `sampling.log_methods` | null | null | Euler is settled; the one place to turn it back on |

**Where the numbers come from:** `./jobs/submit/bench_train.sh` timed the full-data recipe
on the A100 (job 27839620) — 256.6 ms/step as the runs were, 129.9 with train WER off,
98.7 + compile, 70.0 + bf16. Re-run it if the model or batch changes enough to matter.
