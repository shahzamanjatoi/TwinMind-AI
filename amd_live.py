"""Live AMD GPU screening panel for TwinMind AI.

Keep this file next to app.py. If PyTorch is not installed (for example on a
CPU-only public deployment) the app still works and the panel says so.

Scores are standardized with the mean/std saved from the full-dataset GPU run
(reports/amd_gpu/), so a recording is compared with ALL recordings and the
flagged share is meaningful. If those files are missing, the recording is
standardized against itself and the panel warns about it.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
except ImportError:
    torch = None

RESULT_KEY = "live_gpu_result"
TIMING_RUNS = 20
REF_DIR = Path(__file__).resolve().parent / "reports" / "amd_gpu"


def get_gpu_status():
    """Return (gpu_available, device_name, backend_label)."""
    if torch is None:
        return False, None, "PyTorch not installed"
    if not torch.cuda.is_available():
        return False, None, "no GPU detected"
    hip = getattr(torch.version, "hip", None)
    backend = f"ROCm / HIP {hip}" if hip else "CUDA"
    return True, torch.cuda.get_device_name(0), backend


def _load_reference(feature_columns):
    """(center, scale, threshold) from the saved full-dataset run, or None."""
    try:
        summary = pd.read_csv(REF_DIR / "twinmind_gpu_feature_summary.csv")
        summary = summary.set_index("feature").reindex(feature_columns)
        center = summary["mean_after_imputation"].to_numpy(dtype=np.float32)
        scale = summary["standard_deviation"].to_numpy(dtype=np.float32)
        if np.isnan(center).any() or np.isnan(scale).any():
            return None
        screening = pd.read_csv(REF_DIR / "twinmind_gpu_sensor_screening.csv")
        is_flagged = screening["flagged_for_review"].astype(str).str.lower()
        flagged = screening.loc[is_flagged.isin(["true", "1", "yes"]), "screening_score"]
        if flagged.empty:
            return None
        return center, scale, float(flagged.min())
    except Exception:
        return None


def _screen(x_np, device, reference=None):
    """Impute, standardize and score every row; returns (scores, threshold)."""
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    finite = torch.isfinite(x)
    if reference is None:
        means = torch.nanmean(torch.where(finite, x, torch.nan), dim=0)
        fill = torch.nan_to_num(means, nan=0.0)
        x_clean = torch.where(finite, x, fill.unsqueeze(0))
        center = x_clean.mean(dim=0)
        scale = x_clean.std(dim=0, unbiased=False)
    else:
        ref_center, ref_scale, ref_threshold = reference
        center = torch.as_tensor(ref_center, dtype=torch.float32, device=device)
        scale = torch.as_tensor(ref_scale, dtype=torch.float32, device=device)
        x_clean = torch.where(finite, x, center.unsqueeze(0))
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    scores = ((x_clean - center) / scale).abs().mean(dim=1)
    if reference is None:
        return scores, torch.quantile(scores, 0.95)
    return scores, torch.tensor(ref_threshold, dtype=torch.float32, device=device)


def _median_time(x_np, device, reference):
    """Median seconds over several runs, after one warm-up run."""
    _screen(x_np, device, reference)
    times = []
    for _ in range(TIMING_RUNS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _screen(x_np, device, reference)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _run(df, feature_columns, recording_name, gpu_available, gpu_name, backend):
    features = df.reindex(columns=feature_columns).apply(
        pd.to_numeric, errors="coerce"
    )
    x_np = features.to_numpy(dtype=np.float32)
    reference = _load_reference(feature_columns)

    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if gpu_available else None
    run_device = gpu if gpu is not None else cpu

    cpu_time = _median_time(x_np, cpu, reference)
    gpu_time = _median_time(x_np, gpu, reference) if gpu is not None else None

    scores, threshold = _screen(x_np, run_device, reference)
    scores_np = scores.detach().cpu().numpy()
    flagged = int((scores >= threshold).sum().item())
    sample_index = (
        df["sample_index"].to_numpy()
        if "sample_index" in df.columns
        else np.arange(len(df))
    )
    top = (
        pd.DataFrame({"sample_index": sample_index, "screening_score": scores_np})
        .nlargest(5, "screening_score")
        .round(4)
    )

    return {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recording": str(recording_name),
        "rows": int(len(x_np)),
        "features": int(x_np.shape[1]),
        "device_used": gpu_name if gpu is not None else "CPU",
        "backend": backend,
        "standardized_against": (
            "full-dataset reference (reports/amd_gpu)"
            if reference is not None
            else "this recording only"
        ),
        "cpu_median_seconds": cpu_time,
        "gpu_median_seconds": gpu_time,
        "threshold": float(threshold.item()),
        "flagged": flagged,
        "flagged_percent": round(100.0 * flagged / max(1, len(x_np)), 2),
        "top_rows": top,
    }


def render_live_gpu_screening(df, feature_columns, recording_name):
    """Draw the live screening panel inside a Streamlit tab."""
    st.subheader("Live GPU screening")
    st.caption(
        "Runs the statistical screening on the recording selected in the "
        "sidebar, right now, and times it on the CPU and (if present) the "
        "GPU. Timings include copying the data to the device. This is a "
        "screening heuristic, not a fault diagnosis."
    )

    if torch is None:
        st.warning(
            "PyTorch is not installed in this environment, so live "
            "screening is unavailable here."
        )
        return

    available, name, backend = get_gpu_status()
    if available:
        st.success(f"GPU detected: {name} ({backend})")
    else:
        st.info(
            "No GPU detected in this environment, so only the CPU is timed. "
            "Run the app on the AMD notebook to measure the AMD GPU."
        )

    if _load_reference(feature_columns) is None:
        st.warning(
            "Reference files not found in reports/amd_gpu, so this recording "
            "is standardized against itself. In that mode about 5% of rows "
            "are always flagged by construction, which does not indicate "
            "faults."
        )

    if st.button("Run live screening", key="live_gpu_run", type="primary"):
        with st.spinner("Running screening..."):
            try:
                st.session_state[RESULT_KEY] = _run(
                    df, feature_columns, recording_name, available, name, backend
                )
            except Exception as exc:
                st.session_state.pop(RESULT_KEY, None)
                st.error(f"Live screening failed: {exc}")

    result = st.session_state.get(RESULT_KEY)
    if not result:
        return

    st.caption(
        f"Result for {result['recording']} - scores standardized against: "
        f"{result['standardized_against']}; threshold {result['threshold']:.4f}."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows screened", f"{result['rows']:,}")
    c2.metric(
        "Flagged for review",
        f"{result['flagged']:,} ({result['flagged_percent']}%)",
    )
    c3.metric("Device used", str(result["device_used"]))

    t1, t2, t3 = st.columns(3)
    t1.metric("CPU median time", f"{result['cpu_median_seconds'] * 1000:.2f} ms")
    if result["gpu_median_seconds"] is not None:
        gpu_ms = result["gpu_median_seconds"] * 1000
        t2.metric("GPU median time", f"{gpu_ms:.2f} ms")
        t3.metric(
            "CPU time / GPU time",
            f"{result['cpu_median_seconds'] / result['gpu_median_seconds']:.2f}x",
        )
        st.caption(
            f"Measured values, median of {TIMING_RUNS} runs after a warm-up. "
            "On small recordings the CPU can be as fast or faster because "
            "of data-transfer overhead; report whatever you measure."
        )

    st.markdown("**Highest-scoring samples in this recording**")
    st.dataframe(result["top_rows"], hide_index=True)

    export = {k: v for k, v in result.items() if k != "top_rows"}
    st.download_button(
        "Download this run as JSON evidence",
        data=json.dumps(export, indent=2),
        file_name="twinmind_live_gpu_run.json",
        mime="application/json",
        key="live_gpu_download",
    )