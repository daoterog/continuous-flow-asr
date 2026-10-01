"""The interpolant coefficients: alpha, beta and gamma as functions of flow time."""

from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf


def make_schedule(spec: DictConfig | dict | str | None) -> Callable | None:
    """Turn a named coefficient schedule into alpha_fn / beta_fn / gamma_fn.

    The three variants are:
      * `linear_up`   -> t          (beta, the weight on the one-hot target)
      * `linear_down` -> 1 - t      (alpha on the noise; gamma in the audio-conditioned run)
      * `parabola`    -> 4c t (1-t) (gamma in the three-term run, peaking at c)

    None means the term is absent, which is how the linear variant drops mu entirely.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = OmegaConf.to_container(spec) if isinstance(spec, DictConfig) else dict(spec)
    kind = spec.pop("type")

    # `scale` matters most on gamma. At scale 1 the audio_cond schedule has gamma == alpha
    scale = spec.pop("scale", 1.0)

    if kind == "linear_up":
        return lambda t: scale * t
    if kind == "linear_down":
        return lambda t: scale * (1 - t)
    if kind == "parabola":
        c = spec.pop("c", 0.5)
        return lambda t: scale * 4 * c * t * (1 - t)
    raise ValueError(
        f"unknown schedule {kind!r}; expected linear_up, linear_down or parabola"
    )
