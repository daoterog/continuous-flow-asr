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
| `train.compile` | false | false, until the A100 benchmark says otherwise | warmup vs gain |
| `train.precision` | null | null, until the benchmark says otherwise | measured per card |
| `metrics.wer_bins.train` | 10 | 0 | 68.6 vs 0.5 ms of CPU per step; ~15 s of a ~77 s epoch |

| `run/` | sweep | full | why |
|---|---|---|---|
| `train.num_epochs` | 100 | 300 | screen vs final (`TIER` in `jobs/common/sweep.sh`) |
| `sampling.every_n_epochs` | 10 | 5 | a screen is read at its endpoint |
| `sampling.log_methods` | null | null | Euler is settled; the one place to turn it back on |

**Deciding compile and precision:** `./jobs/submit/bench_train.sh` (~130 SBU) times the
full-data recipe on the A100 under this profile — as the runs were, then + compile,
+ bf16 — and runs the best trial twice at once to measure packing. Set the two `snellius`
rows from it, and `SEC_PER_EPOCH` / `PACK_SLOWDOWN` in `jobs/common/budget.sh`.
