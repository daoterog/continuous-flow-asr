# continuous-flow-asr

## Layout

```text
jobs/submit/   you run these        jobs/batch/   sbatch runs these, never you
scripts/       python entry points  open_asr_leaderboard/  pinned submodule (normalizer)
```

## Setup

```bash
uv sync && source .venv/bin/activate     # local
git submodule update --init --recursive
```

Snellius: `./uv_to_conda.sh` then `sbatch jobs/batch/install_env.sh`.
Jobs use `~/.conda/envs/flow_asr` directly — **not** `conda activate`, which silently
no-ops in batch shells.

FFmpeg 7.1.1 is required (torchcodec decodes through it). Check with
`python -m scripts.check_ffmpeg`. On Ubuntu apt caps at 4.4.2, so:

```bash
conda create -y -p ./.ffmpeg -c conda-forge "ffmpeg=7.1.1"
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/../.ffmpeg/lib:$LD_LIBRARY_PATH"' >> .venv/bin/activate
```

## Evaluate

```bash
./jobs/submit/eval_sets.sh      # 1. download ~19 GB + build Arrow cache (2 chained jobs)
LIMIT=32 ./jobs/submit/rehearse.sh   # 2. dress rehearsal, same array shape, 32 utts
./jobs/submit/eval.sh           # 3. real run: 7 datasets, 72k utts, ~1h30
bash -c 'source jobs/batch/_env.sh && python -m scripts.score_eval'   # 4. score
```

Wait for each step. Step 1 emits two job ids; both must finish.
Rehearsal verdict: `cat /scratch-shared/cont-flow-asr/rehearsal/verify_*.out`.

Baseline (`whisper-large-v3`, ours vs leaderboard):

| LS clean | LS other | AMI-C | Giga-C | SPGI | Voxp-C | Earn22-C | avg |
|---|---|---|---|---|---|---|---|
| 1.65 / 1.56 | 3.46 / 3.52 | 13.70 / 13.63 | 8.55 / 8.35 | 2.68 / 2.71 | 4.86 / 4.53 | 8.60 / 7.99 | 6.21 / 6.04 |

## Training features

```bash
./jobs/submit/librispeech.sh                                  # LibriSpeech, 62 GB
./jobs/submit/whisper.sh                                      # encoder latents, trimmed
CONFIGS="clean other" SPLITS=test NO_TRIM=1 ./jobs/submit/whisper.sh   # eval latents
```

`NO_TRIM=1` is required for anything fed back through Whisper's decoder: it
cross-attends over all 1500 frames and emits `<|en|>` repeatedly from a trimmed cache.
Trimmed is 2.4x smaller and fine for training a model that reads latents directly.

## Gotchas

- `load_dataset(repo, config)` resolves the **whole config** before the split — it will
  pull 11 GB for one spgispeech shard. Download with `snapshot_download` + `allow_patterns`.
- Concurrent `load_dataset` calls corrupt the shared Arrow cache on GPFS. Hence the
  single-task cache build in step 1.
- Overrides: `N_JOBS`, `LIMIT`, `DATASETS`, `MODEL`, `WALLTIME`, `CONDA_ENV`.
