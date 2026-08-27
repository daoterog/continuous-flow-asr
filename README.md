# continuous-flow-asr

## Setup

```bash
uv sync
source .venv/bin/activate
```

`torch` and `torchcodec` are pinned to the CUDA 13 index (`cu130`) and must come
from the same one — torchcodec is ABI-locked to torch and the CUDA major version.
Requires an NVIDIA driver >= 580.

### FFmpeg (required)

torchcodec decodes audio through FFmpeg's shared libraries, picking the loader that
matches the installed `libavutil` — the `ffmpeg` binary on `PATH` is not what it
uses. Any major in 4–9 works; pinned to **7.1.1** in `conda_environment.yaml`.

```bash
python -m scripts.check_ffmpeg   # 0 = match, 1 = mismatch, 2 = no pin found
```

- **Snellius:** included in `conda_environment.yaml`. Do *not* `module load FFmpeg` —
  it can shadow conda's libs with a different version.
- **Ubuntu 22.04:** apt caps at 4.4.2, so install into a local prefix and export it
  before python starts (prepended, or the system 4.4.2 wins):

  ```bash
  conda create -y -p ./.ffmpeg -c conda-forge "ffmpeg=7.1.1"
  cat >> .venv/bin/activate <<'EOF'
  export LD_LIBRARY_PATH="$VIRTUAL_ENV/../.ffmpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  EOF
  ```

  `uv sync` can recreate `.venv` and drop that line; re-append it if the check
  starts reporting 4.4.2.

## LibriSpeech

~62 GB across 130 parquet shards. Only `clean` and `other` are reachable — `all` is
the same audio renamed.

```bash
python -m scripts.download_librispeech --configs clean --splits validation
python -m scripts.download_librispeech --dry-run    # plan only
python -m scripts.download_librispeech --verify     # check an existing download
```

`--env snellius` uses `/scratch-shared/cont-flow-asr/data` instead of `./data`.

On the cluster, array tasks split the work by shard (`files[k::N]`), and the sbatch
`--time` is derived from the selection's byte count:

```bash
./jobs/submit_librispeech.sh                            # everything, 8 tasks
SPLITS="validation test" ./jobs/submit_librispeech.sh   # small run first
```

### Reading it

```python
snap = glob.glob("data/hub/datasets--openslr--librispeech_asr/snapshots/*")[0]
ds = load_dataset("parquet", data_files=f"{snap}/clean/validation/*.parquet", split="train")
```

Here `split="train"` is the parquet loader's own name — the LibriSpeech split comes
from the path. Avoid `load_dataset("openslr/librispeech_asr", "clean", split=...)`:
it resolves the config first and pulls every split in it (31 GB) unless already cached.

## Snellius env

```bash
./uv_to_conda.sh                              # writes conda_environment.yaml
conda env create -f conda_environment.yaml
conda activate flow_asr
```
