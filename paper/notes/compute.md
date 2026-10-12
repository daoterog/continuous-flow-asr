GPU-hours are summed from the logged runtime of every training run in the project's experiment
tracker. Cluster runs are on one NVIDIA A100 each; local runs on a workstation GPU. Snellius
bills 128 SBU per GPU-hour and a job at least two GPUs, so a run's cost lies between 1× and 2×
its own hours depending on whether it shared its job. The accounted total also includes what is
not a training run: feature extraction for the dataset, throughput benchmarks, evaluation, and
queueing overhead.
