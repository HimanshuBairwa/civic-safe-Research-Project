"""US multi-channel loader: records + NCVS + 911 calls-for-service.

This is the drop-in point for the ONE real ceiling lever (see
docs/data_access/). OICC needs >=3 MECHANISM-INDEPENDENT channels of the same
latent crime rate. On US data those are:

  channel 0 (pivot) = police-recorded crime      (already in data/processed/*.pt)
  channel 1         = NCVS victimization rate     (survey; captures unreported)
  channel 2         = 911 calls-for-service       (citizen-initiated)

The recorded-crime channel already exists (us_loader.build_us_channels). NCVS and
911 are external (public but not shipped); when you obtain them (templates in
docs/data_access/), drop the aligned files in and this loader assembles the real
3-channel matrix -- at which point the over-identification test and latent
prediction run on GENUINELY independent US channels for the first time.

Until then, `build_us_multichannel(demo=True)` returns a synthetic 3-channel
panel with the correct shape and bias structure so downstream code (and tests)
run today. `build_us_multichannel(records=..., ncvs=..., cfs=...)` takes real
aligned arrays.

ALIGNMENT CONTRACT (what you must provide for the real run):
  - all three channels aligned to the SAME area x period grid (e.g. community
    area x month), same ordering, same length N;
  - each a 1-D array of a NON-NEGATIVE count or rate; the loader applies
    log1p(rate) internally and returns a (3, N) log-channel matrix.
NCVS is coarser than records (national/regional/MSA); you must first small-area-
estimate or broadcast it to the records grid -- see docs/data_access/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ArrayF = np.ndarray


@dataclass
class USMultiChannel:
    log_channels: ArrayF          # (3, N)
    channel_names: list[str]
    n_units: int
    is_demo: bool


def _to_log_rate(x: ArrayF) -> ArrayF:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"each channel must be 1-D; got shape {x.shape}")
    if np.any(~np.isfinite(x)):
        raise ValueError("channel contains non-finite values")
    return np.log1p(np.clip(x, 0.0, None))


def build_us_multichannel(
    records: ArrayF | None = None,
    ncvs: ArrayF | None = None,
    cfs: ArrayF | None = None,
    *,
    demo: bool = False,
    n: int = 3000,
    seed: int = 0,
) -> USMultiChannel:
    """Assemble the real 3-channel US matrix, or a synthetic demo.

    Parameters
    ----------
    records, ncvs, cfs : (N,) non-negative count/rate arrays, aligned to the same
        area x period grid. Provide all three for the real run.
    demo : if True (or if any channel is missing), return a synthetic 3-channel
        panel with a realistic bias structure so downstream code runs today.
    n, seed : synthetic demo size / seed.

    Returns
    -------
    USMultiChannel with a (3, N) log-channel matrix.
    """
    have_real = records is not None and ncvs is not None and cfs is not None
    if have_real and not demo:
        r = _to_log_rate(records)
        v = _to_log_rate(ncvs)
        c = _to_log_rate(cfs)
        lengths = {len(r), len(v), len(c)}
        if len(lengths) != 1:
            raise ValueError(
                f"channels must be aligned to equal length; got "
                f"records={len(r)}, ncvs={len(v)}, cfs={len(c)}"
            )
        mat = np.vstack([r, v, c])
        return USMultiChannel(mat, ["records", "ncvs", "cfs"], len(r), False)

    # synthetic demo: one latent theta measured by three differently-biased
    # channels (records under-reports and is enforcement-biased; NCVS is noisy
    # but near-unbiased; 911 has a different citizen-initiation filter).
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)                    # area covariate (e.g. poverty)
    theta = np.exp(1.2 + 0.5 * x + rng.normal(0.0, 0.5, n))  # true rate
    lt = np.log(theta)
    records_c = lt + (-0.8 - 0.4 * x) + rng.normal(0.0, 0.35, n)  # enforcement bias
    ncvs_c = lt + (0.0) + rng.normal(0.0, 0.55, n)               # ~unbiased, noisy
    cfs_c = lt + (-0.2 + 0.1 * x) + rng.normal(0.0, 0.40, n)     # citizen filter
    mat = np.vstack([records_c, ncvs_c, cfs_c])
    return USMultiChannel(mat, ["records", "ncvs(demo)", "cfs(demo)"], n, True)


def load_real_if_available(processed_dir: str | Path) -> USMultiChannel | None:
    """Load a real 3-channel run if aligned files exist, else None.

    Looks for `us_records.npy`, `us_ncvs.npy`, `us_cfs.npy` (1-D, aligned) under
    `processed_dir`. This is the single place to wire real data: drop those three
    files in (see docs/data_access/) and every downstream experiment picks them up.
    """
    d = Path(processed_dir)
    paths = {k: d / f"us_{k}.npy" for k in ("records", "ncvs", "cfs")}
    if not all(p.exists() for p in paths.values()):
        return None
    arrs = {k: np.load(p) for k, p in paths.items()}
    return build_us_multichannel(arrs["records"], arrs["ncvs"], arrs["cfs"])
