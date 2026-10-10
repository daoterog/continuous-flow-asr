# Jobs on Snellius

`submit/` sizes and submits, `batch/` is what runs on the node, `sweeps/` builds sets of
runs, and `common/` holds what they share.

## Budget standards

The budget is **100k SBU counted from 25-08-2026**. A training job bills **256 SBU per hour**
(one A100 plus its quarter of the node) for the time it actually runs, so a full-data
300-epoch run costs ~1,730 SBU. One run alone keeps the GPU at a median 39%, so two runs
share each job.

| tier | epochs | seeds | runs per GPU job | expected per job | use |
|---|---|---|---|---|---|
| `screen` | 100 | 1 | 2 | ~900 SBU | the default: rank the arms |
| `confirm` | 300 | 2 | 2 (the seeds share a job) | ~2,600 SBU | only the screen's winner(s) |

A screen carries its own control: a 100-epoch run is only comparable to another one.

**Rules**
- **Gate:** every submission prints its expected SBU and is checked against the cap by
  `common/budget.sh`. It counts what has been spent (`accuse`), what queued jobs will still
  cost, and the submission itself.
  - Over the cap is refused with no override.
  - A sweep over 3,000 SBU is refused unless `BUDGET_OK=1`.
- **Ledger:** every job is appended to `jobs/ledger.tsv` (gitignored) with its expected cost.
- **Walltime:** generous is free, since billing is on elapsed time. Sweeps ask for twice the
  estimate; a run cut off before it finishes is the expensive failure.
- **Before planning a sweep:** run `./jobs/budget.sh`. It shows spent / queued / left and
  the last ledger rows.

**Writing a sweep:** source `common/sweep.sh`, call `tier_defaults`, `queue_run` once per
run, then `flush_runs`. See `sweeps/mu_reg.sh`. Without `SUBMIT=1` it prints the plan and
its SBU and submits nothing.

**Profiles:** what runs on Snellius gets `--where snellius` (data root, workers, eval
batch, cheap per-step metrics) and the tier's `--run sweep|full` (epochs, eval cadence) —
see `configs/profiles/README.md`. Sweep `COMMON` blocks hold the recipe only.

**Calibration (pending):** `SEC_PER_EPOCH=80` and `PACK_SLOWDOWN=1.5` in `common/budget.sh`
are from the runs before the profiles. Run `./jobs/submit/bench_train.sh` (~130 SBU) once:
part 1 gives the single-run ms/step and GPU % per setting, part 2 runs the best setting
twice at once. Set `SEC_PER_EPOCH` from part 1's winner, `PACK_SLOWDOWN` = single utts/s ÷
packed utts/s per run, and keep `PACK=2` only if that stays well below 2.

**Packing by hand:** separate runs with `::` in `submit/train.sh`, e.g.
`./jobs/submit/train.sh configs/audio_cond.yaml seed=0 :: seed=1`. Each run logs to
`slurm_output_<job>_<i>.out`.
