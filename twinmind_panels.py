"""Extra TwinMind panels, imported by app.py.

- render_gpu_session_details: AMD GPU run details read from saved JSON
  (replaces the hard-coded numbers in the AMD GPU tab).
- render_temporal_evaluation: chronological-holdout results of the PyTorch
  temporal model (Evaluation tab).
- sensor_figure: readable Sensor Explorer charts for sensors on very
  different scales.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SENSOR_LAYOUTS = (
    "One panel per sensor",
    "Overlay (z-score)",
    "Overlay (raw values)",
)

_SUMMARY_ROWS = ("accuracy", "macro avg", "weighted avg")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def _pretty(name):
    return str(name).replace("_", " ").title()


def _ms(seconds):
    return f"{float(seconds) * 1000:,.2f} ms"


def _zscore(series):
    centered = series - series.mean()
    spread = series.std()
    return centered / spread if pd.notna(spread) and spread > 0 else centered


# ------------------------------------------------------------
# Sensor Explorer
# ------------------------------------------------------------

def sensor_figure(df, sensors, layout):
    """Return (figure, height) for the selected sensors and layout."""
    sensors = list(sensors)
    long = df[["sample_index"] + sensors].melt(
        id_vars="sample_index", var_name="Sensor", value_name="Value"
    )
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")

    if layout == SENSOR_LAYOUTS[0]:
        spacing = min(0.04, 1 / max(len(sensors) - 1, 1) - 0.001)
        fig = px.line(
            long,
            x="sample_index",
            y="Value",
            facet_row="Sensor",
            category_orders={"Sensor": sensors},
            facet_row_spacing=spacing,
            title="Sensor signals across samples (independent y-axis per sensor)",
        )
        fig.update_yaxes(matches=None, showticklabels=True, title_text="")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(showlegend=False)
        return fig, max(320, 190 * len(sensors))

    if layout == SENSOR_LAYOUTS[1]:
        long["Value"] = long.groupby("Sensor")["Value"].transform(_zscore)
        fig = px.line(
            long,
            x="sample_index",
            y="Value",
            color="Sensor",
            title="Sensor signals across samples (z-score per sensor)",
            labels={"Value": "z-score"},
        )
        return fig, 480

    fig = px.line(
        long,
        x="sample_index",
        y="Value",
        color="Sensor",
        title="Sensor signals across samples (raw values)",
    )
    return fig, 480


# ------------------------------------------------------------
# AMD GPU tab
# ------------------------------------------------------------

def render_gpu_session_details(amd_dir):
    """Show the saved AMD GPU run details and GPU-vs-CPU timing."""
    amd_dir = Path(amd_dir)
    evidence = _read_json(amd_dir / "gpu_session_evidence.json")
    timing = _read_json(amd_dir / "temporal_inference_timing.json")

    st.markdown("**AMD workload details**")
    if evidence is None:
        st.warning(
            "`gpu_session_evidence.json` was not found in "
            f"`{amd_dir.name}`, so the run details cannot be shown."
        )
    else:
        rows = evidence.get("rows")
        before = evidence.get("rows_before_dedupe")
        st.write(f"GPU: {evidence.get('gpu', 'n/a')}")
        st.write(
            f"Software: PyTorch {evidence.get('pytorch', 'n/a')} "
            f"with ROCm / HIP {evidence.get('hip', 'n/a')}"
        )
        if evidence.get("run_at"):
            st.write(f"Run at (UTC): {str(evidence['run_at'])[:19].replace('T', ' ')}")
        if rows is not None and before is not None:
            st.write(
                f"Rows screened: {rows:,} after removing duplicates "
                f"({before:,} before)"
            )
        if evidence.get("excluded_files"):
            st.write(
                "Excluded from screening: "
                + ", ".join(f"`{name}`" for name in evidence["excluded_files"])
            )
        if evidence.get("threshold") is not None:
            st.write(f"Screening threshold: {float(evidence['threshold']):.4f}")
        if evidence.get("flagged") is not None and rows:
            st.write(
                f"Flagged for review: {int(evidence['flagged']):,} "
                f"({100 * evidence['flagged'] / rows:.2f}%)"
            )
        if evidence.get("missing_or_non_finite") is not None:
            st.write(
                f"The original input contained {int(evidence['missing_or_non_finite']):,} "
                "missing or non-finite sensor values before imputation."
            )

        cpu = evidence.get("cpu_median_seconds")
        gpu = evidence.get("gpu_median_seconds")
        if cpu and gpu:
            m1, m2, m3 = st.columns(3)
            m1.metric("CPU median (screening)", _ms(cpu))
            m2.metric("AMD GPU median (screening)", _ms(gpu))
            m3.metric("GPU speed-up (screening)", f"{cpu / gpu:.1f}x")
            st.caption(
                f"Median of {evidence.get('timing_runs', 'several')} timed runs of "
                "the screening computation. It is a small workload, so the "
                "speed-up is modest; the temporal model below is the heavier one."
            )

    if timing is None:
        return

    st.markdown("**Temporal model inference: AMD GPU vs CPU**")
    gpu_seconds = timing.get("gpu_median_seconds")
    best_cpu = timing.get("best_cpu_median_seconds")
    if gpu_seconds and best_cpu:
        t1, t2, t3 = st.columns(3)
        t1.metric("AMD GPU median (temporal model)", _ms(gpu_seconds))
        t2.metric("Best CPU median (temporal model)", _ms(best_cpu))
        t3.metric("GPU speed-up vs best CPU", f"{best_cpu / gpu_seconds:.1f}x")

        by_threads = timing.get("cpu_median_by_threads_seconds") or {}
        bars = [
            {"Device": f"CPU, {threads} thread(s)", "Median time (ms)": seconds * 1000}
            for threads, seconds in by_threads.items()
        ]
        bars.append({"Device": "AMD GPU", "Median time (ms)": gpu_seconds * 1000})
        fig = px.bar(
            pd.DataFrame(bars),
            x="Device",
            y="Median time (ms)",
            title=f"Median inference time on {int(timing.get('rows', 0)):,} rows",
        )
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    notes = []
    default_cpu = timing.get("cpu_median_seconds")
    if default_cpu and gpu_seconds:
        notes.append(
            f"With the default {timing.get('cpu_threads_default', 'n/a')}-thread "
            f"CPU setting ({_ms(default_cpu)}) the ratio would be "
            f"{default_cpu / gpu_seconds:.1f}x, but that setting was slower than "
            "fewer threads, so the tuned CPU time above is the fair comparison."
        )
    agree = (timing.get("label_columns_agree") or {}).get("predicted_status")
    if agree is not None:
        notes.append(f"Predicted labels agreed on {agree * 100:.1f}% of rows")
    if timing.get("max_abs_diff_numeric") is not None:
        notes.append(
            f"largest numeric difference between CPU and GPU outputs: "
            f"{timing['max_abs_diff_numeric']:.1e}."
        )
    if notes:
        st.caption(" ".join(notes))


# ------------------------------------------------------------
# Evaluation tab
# ------------------------------------------------------------

def render_temporal_evaluation(temporal_dir):
    """Show the chronological-holdout evaluation of the temporal model."""
    temporal_dir = Path(temporal_dir)
    meta = _read_json(temporal_dir / "experiment_metadata.json")
    if meta is None:
        st.info(
            "Temporal-model results were not found "
            f"(expected in `{temporal_dir.name}`)."
        )
        return

    st.subheader("Chronological holdout evaluation (PyTorch temporal model)")
    st.caption(
        f"Experiment: {meta.get('experiment', 'n/a')}. "
        f"Dataset: `{meta.get('dataset', 'n/a')}`. "
        f"Trained and evaluated on {meta.get('gpu', 'an AMD GPU')}."
    )

    classes = meta.get("classes", [])
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Test records", f"{int(meta.get('test_records', 0)):,}")
    e2.metric("Accuracy", f"{meta.get('accuracy', 0) * 100:.2f}%")
    e3.metric("Macro F1", f"{meta.get('macro_f1', 0):.3f}")
    e4.metric("Classes", len(classes))

    matrix = _read_csv(temporal_dir / "confusion_matrix.csv", index_col=0)
    if matrix is not None:
        matrix.index = [_pretty(v) for v in matrix.index]
        matrix.columns = [_pretty(v) for v in matrix.columns]
        fig = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            title="Confusion matrix: chronological holdout",
            labels={"x": "Predicted class", "y": "True class", "color": "Records"},
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, width="stretch")

    report = _read_csv(temporal_dir / "classification_report.csv", index_col=0)
    if report is not None:
        per_class = report.drop(
            index=[i for i in _SUMMARY_ROWS if i in report.index], errors="ignore"
        )
        if "recall" in per_class.columns and not per_class.empty:
            weakest = per_class["recall"].idxmin()
            st.info(
                f"Lowest recall: {_pretty(weakest)} "
                f"({per_class.loc[weakest, 'recall'] * 100:.1f}%). "
                "The confusion matrix shows which classes those readings "
                "were assigned to."
            )
        report.index = [_pretty(i) if i not in _SUMMARY_ROWS else i for i in report.index]
        st.subheader("Classification report (holdout)")
        st.dataframe(report.round(3), width="stretch")

    mistakes = _read_csv(temporal_dir / "misclassified_readings.csv")
    if mistakes is not None and not mistakes.empty:
        pairs = (
            mistakes.groupby(["true_status", "predicted_status"])
            .size()
            .reset_index(name="Records")
            .sort_values("Records", ascending=False)
        )
        pairs["true_status"] = pairs["true_status"].map(_pretty)
        pairs["predicted_status"] = pairs["predicted_status"].map(_pretty)
        pairs = pairs.rename(
            columns={"true_status": "True class", "predicted_status": "Predicted as"}
        )
        st.markdown(f"**Misclassified readings: {len(mistakes):,}**")
        st.dataframe(pairs, hide_index=True, width="stretch")
        with st.expander("Preview misclassified readings"):
            st.dataframe(mistakes.head(50), hide_index=True, width="stretch")
        st.download_button(
            "Download misclassified readings (CSV)",
            data=mistakes.to_csv(index=False).encode("utf-8"),
            file_name="twinmind_misclassified_readings.csv",
            mime="text/csv",
            key="download_temporal_misclassified",
        )

    if meta.get("features"):
        with st.expander("Features used by the temporal model"):
            for feature in meta["features"]:
                st.write(f"- `{feature}`")

    st.warning(
        "Evaluation limitation: the training and test readings both come from "
        f"one file (`{meta.get('dataset', 'n/a')}`), split chronologically. "
        "This shows temporal generalization within that recording, not "
        "performance on a different motor or an independent test run."
    )


# ------------------------------------------------------------
# Maintenance assistant
# ------------------------------------------------------------

def _assistant_step_preview(result):
    if not isinstance(result, dict):
        return str(result)
    if result.get("error"):
        return str(result["error"])
    if "recordings" in result:
        return f"{len(result['recordings'])} recordings"
    if "flagged" in result:
        return (
            f"{result.get('recording')}: {result.get('flagged')} flagged "
            f"({result.get('flagged_percent')}%)"
        )
    if "condition" in result:
        return (
            f"{result.get('condition')}: recall {result.get('recall')}, "
            f"precision {result.get('precision')}"
        )
    if "classes" in result:
        return f"accuracy {result.get('accuracy')}; weakest {result.get('weakest_recall')}"
    matches = result.get("matches") or []
    if matches:
        return ", ".join(item.get("file", "") for item in matches)
    return json.dumps(result)[:400]


def _render_assistant_trace(trace):
    for step in trace or []:
        label = step.get("tool", "tool")
        with st.status(f"Tool: {label}", state="complete", type="step"):
            args = step.get("args") or {}
            if args:
                st.caption(", ".join(f"{k}={v}" for k, v in args.items()))
            st.write(_assistant_step_preview(step.get("result")))


def render_maintenance_assistant(recording_name, gpu_available, gpu_name, gpu_backend):
    """Chat over saved AMD screening results and maintenance documents."""
    from agent import plan_and_answer

    st.subheader("Maintenance assistant")
    st.caption(
        "Answers come from saved AMD ROCm screening files, the temporal GPU "
        "evaluation, and the local inspection notes. The assistant does not "
        "invent numbers and does not confirm a physical fault."
    )

    compute = st.container(horizontal=True, gap="small")
    with compute:
        if gpu_available:
            st.badge(f"Live GPU: {gpu_name}", color="green")
        else:
            st.badge(f"Live compute: CPU ({gpu_backend})", color="gray")
        st.badge("Assistant: saved AMD reports", color="blue")
        st.badge("Human review required", color="orange")

    st.session_state.setdefault("assistant_messages", [])
    messages = st.session_state.assistant_messages

    suggestions = {
        f"Briefing for {recording_name}": (
            f"Give me a short briefing for {recording_name}: flagged share and what to inspect."
        ),
        "Eccentric rotor reliability": "How reliable is the eccentric rotor prediction?",
        "Faulted bearing checks": "What should I check first for a faulted bearing?",
        "Available recordings": "Which recordings do you have results for?",
    }

    pending = None
    if not messages:
        selected = st.pills(
            "Try asking",
            list(suggestions.keys()),
            label_visibility="collapsed",
        )
        if selected:
            pending = suggestions[selected]

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                with st.status(
                    f"Tools in {message.get('seconds', 0)} s",
                    state="complete",
                    type="compact",
                ):
                    _render_assistant_trace(message.get("trace"))
            st.markdown(message["content"])

    typed = st.chat_input(
        "Ask about flags, reliability, or inspection steps",
        submit_mode="disable",
    )
    if typed:
        pending = typed

    if pending:
        messages.append({"role": "user", "content": pending})
        with st.chat_message("user"):
            st.markdown(pending)
        with st.chat_message("assistant"):
            with st.status(":shimmer[Reading TwinMind files]", type="compact") as status:
                result = plan_and_answer(pending, recording_hint=recording_name)
                _render_assistant_trace(result.get("trace"))
                status.update(
                    label=f"Tools in {result['seconds']} s",
                    state="complete",
                )
            st.markdown(result["answer"])
        messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "trace": result.get("trace"),
                "seconds": result.get("seconds"),
                "backend": result.get("backend"),
            }
        )