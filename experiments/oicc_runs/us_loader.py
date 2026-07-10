"""US real-data testbed: Chicago / NYC crime-category channels for OICC.

This is deliberately a CONTRASTING testbed to the India NCRB run. The US panels
give recorded crime by CATEGORY (violent / property / drug) at area x week. These
three categories all pass through the SAME police recording filter, so they are
NOT mechanism-independent channels of a single latent -- they share the recording
process. We therefore EXPECT the over-identification test to behave differently
than on India's institutionally-independent channels.

Scientific point (stated honestly): treating same-filter crime categories as
"channels" is a NEGATIVE CONTROL for the method itself. If the over-ID test
rejected on India's cross-institution channels it would worry us; that it does
NOT reject there, while the assumptions are visibly strained here, is the kind of
contrast that supports (rather than proves) the method. We report the structure
and the over-ID verdict; we do not claim a latent victimization channel from
crime categories alone.

Channels used (log per-area rate, aggregated over the weekly axis to a coarser
period to stabilize moments):
  C0 = violent    C1 = property    C2 = drug     (+ optional lag channel)

Loads the pre-processed panels shipped in data/processed/{chicago,nyc}_panel.pt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_panel(path: Path):
    try:
        import torch  # local import so the package doesn't hard-depend on torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "loading US .pt panels requires torch; install it "
            "(pip install torch) or run the synthetic/India paths only"
        ) from exc
    p = torch.load(path, weights_only=False)
    counts = np.asarray(p["counts"], dtype=float)   # (S, T, C)
    meta = p["metadata"]
    return counts, meta


def build_us_channels(
    panel_path: str | Path,
    period_weeks: int = 4,
) -> dict:
    """Assemble crime-category channels from a US panel as an OICC input.

    Parameters
    ----------
    panel_path : path to a *_panel.pt file (torch dict with 'counts' (S,T,C)).
    period_weeks : aggregate consecutive weeks into periods of this length to
        stabilize the moment estimates (more counts per cell).

    Returns
    -------
    dict with:
      log_channels : (C, S*P) log per-period category rates (channels = categories)
      channel_names : list[str] category names
      n_units : int number of area-period cells
    """
    panel_path = Path(panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"panel not found: {panel_path}")
    counts, meta = _load_panel(panel_path)
    S, T, C = counts.shape
    cats = list(meta.get("categories", [f"cat{c}" for c in range(C)]))

    P = T // period_weeks
    if P < 4:
        raise ValueError(f"too few periods ({P}); reduce period_weeks")
    # aggregate weeks -> periods
    agg = counts[:, : P * period_weeks, :].reshape(S, P, period_weeks, C).sum(axis=2)
    # log(1 + count) per area-period, one row per category (channel)
    # flatten area x period into units
    chan = np.empty((C, S * P), dtype=float)
    for c in range(C):
        chan[c] = np.log1p(agg[:, :, c].reshape(-1))

    return {
        "log_channels": chan,
        "channel_names": cats,
        "n_units": S * P,
        "n_areas": S,
        "n_periods": P,
    }
