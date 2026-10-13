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
| `data.audio_stride` | 4 | 1 | full 1500-frame latents on the cluster, always; pooling is a laptop-only shortcut |
| `data.batch_size` | base's 32 | 128 | the knee of the stride-1 ladder for small + micro (96% of the best), and skilled-lake-22's batch, so lr 2e-3 carries over |
| `data.gpu_prefetch` | (off) | true | copy the next ~490 MB batch while the current one trains: 643 → 736 utts/s |
| `data.num_workers` | 4 | 17 | 6-core laptop vs 18 cores per run; the 17th worker measured +12% |
| `train.matmul_precision` | highest | high | TF32 for the fp32 output projection: +6% on the A100 |
| `data.eval_batch_size` | null | 512 | 6 GB vs 40 GB card |
| `schedule.probe_n_utts` | 2048 | 8192 | probe cost vs precision |
| `train.compile` | false | true | 129.9 → 98.7 ms/step on the A100; warmup wasted on short local runs |
| `train.precision` | null | bf16-mixed | 98.7 → 70.0 ms/step; compare bf16 runs only with bf16 runs |
| `metrics.wer_bins.train` | 10 | 0 | 68.6 vs 0.5 ms of CPU per step; ~15 s of a ~77 s epoch |

| `run/` | sweep | full | why |
|---|---|---|---|
| `train.num_epochs` | 100 | 300 | screen vs final (`TIER` in `jobs/common/sweep.sh`) |
| `train.check_val_every_n_epoch` | 10 | 10 | measured: val pass 10.7 s + probe 22.2 s vs ~54 s of training per epoch → ~6% |
| `sampling.every_n_epochs` | 10 | 10 | must equal the val cadence: the probe runs inside validation, and sampled WER picks the checkpoints |
| `sampling.log_methods` | null | null | Euler is settled; the one place to turn it back on |

**Where the numbers come from:** `./jobs/submit/bench_train.sh` timed the full-data recipe
on the A100 (job 27839620) — 256.6 ms/step as the runs were, 129.9 with train WER off,
98.7 + compile, 70.0 + bf16. Re-run it if the model or batch changes enough to matter.

## Recipes vs profiles

A **recipe** (`configs/recipes/*.yaml`) says *what* is trained: the model sizes, the data
splits, the LRs, the schedule. A **profile** says *where* it runs and *how long*. On Snellius,
`train.build_config` refuses a config without `recipe:`, whether it was reached with
`--where snellius` or with a plain `data.env=snellius` override. That's because a bare variant
silently inherits base.yaml's laptop model; every cluster run up to 2026-10-10 did. A
deliberate one-off passes `recipe=<name>`.

A recipe must never set a key a profile owns (the tables above). The recipe layer sits above
the profiles, so it would win silently: `flow_mapping.yaml`'s `num_epochs: 50` once made
`--run full` train 50 epochs while the sweep priced 300. Only pin such a key to override a
profile on purpose (`flow_mapping.yaml` keeps `train.compile: false`).
