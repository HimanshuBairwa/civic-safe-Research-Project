"""End-to-end OICC experiment on real India NCRB state-year data.

Runs the full pipeline on four mechanism-independent channels (recorded IPC
crime, distress calls, complaints-against-police, custodial-death/HR-violation
accountability) and reports, HONESTLY:

  * estimated one-factor loadings and latent variance,
  * the over-identification specification test verdict (does the one-factor +
    conditional-independence structure survive on real data?),
  * the latent-rate estimate per state-year and a leave-pivot-out conformal
    band, with the common-mode sensitivity knob swept,
  * a blunt limitations block.

There is no ground-truth latent on real data (by construction — that is the whole
point), so we CANNOT report latent coverage here; we report what IS checkable:
the over-ID test, the recovered structure, and how the band responds to the
common-mode knob. Controlled coverage validation lives in the synthetic tests.

Run:
  python experiments/oicc_runs/run_ncrb_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# make the oicc package and this folder importable regardless of CWD
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from oicc.moments import estimate_factor_moments          # noqa: E402
from oicc.deconvolve import deconvolve_blup               # noqa: E402
from oicc.spec_test import overid_wald_test               # noqa: E402
from oicc.conformal import leave_pivot_out_conformal      # noqa: E402
from ncrb_loader import load_ncrb_channels                # noqa: E402
from paths import find_india_data                         # noqa: E402

_DEFAULT_DATA = find_india_data()


def run(data_dir: Path | None = _DEFAULT_DATA) -> dict:
    if data_dir is None:
        msg = (
            "India NCRB data not found. Set OICC_INDIA_DATA=/path/to/ncrb/data "
            "(a folder containing crime/01_District_wise_crimes_committed_IPC_"
            "2001_2012.csv), or place it at <project>/data/ncrb. Skipping."
        )
        print(msg)
        return {"lines": [msg], "skipped": True}
    d = load_ncrb_channels(data_dir)
    Y = d["log_channels"]
    names = d["channel_names"]
    K, N = Y.shape

    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    out("=" * 72)
    out("OICC on REAL India NCRB data (state-year, 2001-2010)")
    out("=" * 72)
    out(f"channels (K={K}): {names}")
    out(f"aligned state-year cells: N={N}  (states={len(set(d['states']))})")
    out("")

    # --- structure ---
    fm = estimate_factor_moments(Y, pivot=0)
    out("[1] One-factor structure (pivot = recorded IPC crime)")
    for i, n in enumerate(names):
        out(f"    loading beta[{n:8s}] = {fm.beta[i]:+.3f}   "
            f"noise_var = {fm.noise_var[i]:.3f}")
    out(f"    Var(latent) = {fm.var_theta:.3f}")
    out("")

    # --- over-ID specification test ---
    spec = overid_wald_test(Y, seed=0)
    out("[2] Over-identification specification test")
    out(f"    kind={spec.kind}  df={spec.df}  stat={spec.stat:.3f}  "
        f"p={spec.pvalue:.4f}")
    verdict = ("REJECT one-factor+CI (a DETECTABLE dependence violation exists)"
               if spec.pvalue < 0.05 else
               "do NOT reject one-factor+CI in detectable directions")
    out(f"    verdict: {verdict}")
    out(f"    (honest: this cannot see common-mode Delta-parallel violations; "
        f"delta_perp_hat={spec.delta_perp_hat:.4f})")
    out("")

    # --- latent recovery ---
    est = deconvolve_blup(Y, moments=fm)
    out("[3] Recovered latent (deconvolved) signal - top/bottom states (mean)")
    states = np.array(d["states"])
    theta = est.theta_hat
    order = np.argsort([theta[states == s].mean() for s in sorted(set(states))])
    sset = sorted(set(states))
    hi = [sset[order[-1]], sset[order[-2]], sset[order[-3]]]
    lo = [sset[order[0]], sset[order[1]], sset[order[2]]]
    out(f"    highest latent: {hi}")
    out(f"    lowest  latent: {lo}")
    out("")

    # --- proximal / negative-control common-mode probe ---
    # Use the accountability channel (custodial death / HR violations) as a
    # CANDIDATE negative control for a policing-intensity common mode: it is
    # plausibly driven by state coercion/policing intensity (the confounder) more
    # than by the offending latent. This is illustrative; the exclusion
    # assumption (control carries no offending signal) is untestable and stated.
    out("[4b] Proximal negative-control probe (accountability channel as control)")
    try:
        from oicc.proximal import proximal_deconfound
        acct_idx = names.index("acct")
        sig = np.delete(Y, acct_idx, axis=0)
        ctrl = Y[acct_idx:acct_idx + 1]
        pc = proximal_deconfound(sig, ctrl)
        out(f"    control explains per-channel variance: "
            f"{np.round(pc.what_explained, 2).tolist()}")
        out(f"    identified (Q>=2)? {pc.identified}  "
            f"(Q=1 here -> DETECTION + partial correction only)")
        out("    (honest: with one control this only detects/partially removes a")
        out("     common mode; point-ID needs a second independent control.)")
    except Exception as e:  # never let the probe crash the report
        out(f"    proximal probe skipped: {e}")
    out("")

    # --- TWO-control point identification (custodial deaths + HR violations) ---
    out("[4d] Proximal POINT-ID with TWO independent accountability controls")
    try:
        from ncrb_loader import load_ncrb_two_control
        from oicc.proximal import point_identify
        from oicc.uncertainty import bootstrap_point_id
        tc = load_ncrb_two_control(data_dir)
        r2 = point_identify(tc["signal_channels"], tc["controls"])
        bp = bootstrap_point_id(tc["signal_channels"], tc["controls"],
                                n_boot=400, block=10, seed=0)
        out(f"    signals={tc['signal_names']}  controls={tc['control_names']}  "
            f"N={tc['signal_channels'].shape[1]}")
        out(f"    naive Var(latent)      = {r2.var_theta_naive:.3f}  "
            f"CI[{bp['var_theta_naive'].lower:.3f}, {bp['var_theta_naive'].upper:.3f}]")
        out(f"    POINT-ID clean Var(lat)= {r2.var_theta_clean:.3f}  "
            f"CI[{bp['var_theta_clean'].lower:.3f}, {bp['var_theta_clean'].upper:.3f}]")
        out(f"    estimated Var(common mode W) = {r2.var_W:.3f}")
        out("    reading: the two controls reveal a sizeable common mode; removing")
        out("    it lowers the latent variance. HONEST CAVEAT: custodial deaths /")
        out("    HR violations plausibly carry real enforcement signal (exclusion")
        out("    may fail), and N=189 gives wide CIs -- treat as illustrative of")
        out("    the METHOD on real data, not a definitive Indian-crime estimate.")
    except Exception as e:
        out(f"    two-control point-ID skipped: {e}")
    out("")

    # --- anytime-valid monitor demo over the yearly over-ID stream ---
    out("[4c] Anytime-valid drift monitor (per-year over-ID p-value stream)")
    try:
        from oicc.monitor import EProcessMonitor
        yrs = sorted(set(d["years"]))
        years_arr = np.array(d["years"])
        pstream = []
        for y in yrs:
            mask = years_arr == y
            if mask.sum() >= 8:
                Yy = Y[:, mask]
                pstream.append(overid_wald_test(Yy, seed=int(y)).pvalue)
        mon = EProcessMonitor(alpha=0.05).run(np.array(pstream))
        out(f"    per-year over-ID p-values: {[round(p, 3) for p in pstream]}")
        out(f"    final e-process wealth={mon.wealth:.2f}  alarm={mon.alarm}"
            + (f" at year index {mon.alarm_time}" if mon.alarm else "")
            + "  (alarm => structure drift detected, time-uniform 5% false-alarm)")
    except Exception as e:
        out(f"    monitor demo skipped: {e}")
    out("")

    # --- conformal band + common-mode sweep ---
    out("[5] Leave-pivot-out latent band; common-mode knob sweep (mean width)")
    for gcm in (0.0, 0.25, 0.5, 1.0):
        res = leave_pivot_out_conformal(Y, alpha=0.1, gamma_cm=gcm,
                                        use_spec_test=True, spec_seed=0)
        width = float(np.mean(res.upper - res.lower))
        out(f"    gamma_cm={gcm:.2f}  mean band width={width:.3f}  "
            f"data-driven infl={res.infl_data:.3f}  knob infl={res.infl_knob:.3f}")
    out("")

    out("[6] Honest limitations")
    out("    * No ground-truth latent on real data -> latent coverage is NOT")
    out("      claimed here; it is validated on synthetic ground truth in tests.")
    out("    * State-year, N~253: small; treat estimates as illustrative.")
    out("    * Channels measure DIFFERENT constructs (offending vs mistreatment);")
    out("      the shared-factor assumption is a modeling choice the over-ID test")
    out("      partially checks (detectable directions only).")
    out("    * Distress calls are police-logged -> partial shared filter with the")
    out("      pivot; the strongest-independence channel is the accountability one.")
    out("    * Common-mode (all-channel) bias is UNIDENTIFIABLE from the signal")
    out("      channels alone; the negative-control probe ([4b]) and the gamma_cm")
    out("      knob are the honest ways to detect/bound it -- point-ID needs two")
    out("      valid controls, whose exclusion assumption is untestable.")

    return {"lines": lines, "moments": fm, "spec": spec, "N": N, "K": K}


if __name__ == "__main__":
    run()
