# Jobs on Snellius

`submit/` sizes and submits, `batch/` is what runs on the node, `sweeps/` builds sets of
runs, and `common/` holds what they share.

## Budget standards

The budget is **100k SBU counted from 25-08-2026**. An A100 hour is **128 SBU**, but a job
is billed for the largest of its node shares (1 GPU, 18 cores or 120 GiB each). A training
job needs 160G for the feature cache, which is 2 shares, so **every job is billed as 2
GPUs**. Jobs therefore carry two runs, each on its own GPU, for the price of one.

Cluster runs train `configs/recipes/fulldata.yaml` (denoiser small + mu micro, full
1500-frame latents) under the snellius profile: compile + bf16, batch 128, the next batch
prefetched to the GPU, 17 loader workers, TF32 matmuls, validation and probing every 10
epochs. Measured on the A100 (2026-10-11) that is ~710–840 utts/s, ~45 s per epoch on a
slow node.

| tier | epochs | seeds | runs per job (one GPU each) | expected per job | use |
|---|---|---|---|---|---|
| `screen` | 100 | 1 | 2 | ~363 SBU | the default: rank the arms |
| `confirm` | 300 | 2 | 2 (the seeds share a job) | ~1,003 SBU | only the screen's winner(s) |

These rest on `SEC_PER_EPOCH=55` in `common/budget.sh`. Re-set it from a screen's real
epoch times if they differ.

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

**Throughput:** `./jobs/submit/bench_train.sh` (~128 SBU) re-times the recipe on the A100
under each setting, if the model or the batch changes enough to matter.

**Pairing by hand:** separate runs with `::` in `submit/train.sh`, e.g.
`./jobs/submit/train.sh configs/audio_cond.yaml seed=0 :: seed=1`. Each run gets its own GPU
and logs to `slurm_output_<job>_<i>.out`. A lone run costs the same as a pair, so pair
whenever there is a second run to give it.
