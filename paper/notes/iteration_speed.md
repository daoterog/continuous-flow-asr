The ablations above amount to a few hundred training runs. That was only affordable because each
one was made cheap. Locally, the input pipeline and the metrics were profiled and rewritten,
and two runs were packed per GPU. On the cluster, the workflow settled into fixed tiers:

- a 100-epoch, one-seed *screen* ranks candidate changes;
- a 300-epoch, two-seed *confirm* is reserved for winners;
- two runs are packed into every job, which is billed for two GPUs either way;
- every sweep is priced and gated against the compute budget before submission.

Configurations come from versioned recipes, so a screen trains exactly the model it names.
