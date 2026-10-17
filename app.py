"""TwinMind AI - motor condition monitoring and fault classification (Streamlit)."""

import json
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# The AMD live panel needs PyTorch. If it cannot be imported (for example on a
# CPU-only host), the rest of the app must still work.
try:
    from amd_live import get_gpu_status, render_live_gpu_screening
except Exception:  # noqa: BLE001

    def get_gpu_status():
        return False, "", "amd_live module unavailable"

    def render_live_gpu_screening(*args, **kwargs):
        st.info(
            "Live GPU screening is unavailable on this host. "
            "See the saved AMD results below."
        )


from twinmind_panels import (
    SENSOR_LAYOUTS,
    render_gpu_session_details,
    render_maintenance_assistant,
    render_temporal_evaluation,
    sensor_figure,
)

from twinmind_theme import inject_theme, render_hero

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_TITLE = "TwinMind AI"
BASE_DIR = Path(__file__).resolve().parent
AMD_DIR = BASE_DIR / "reports" / "amd_gpu"
TEMPORAL_DIR = BASE_DIR / "reports" / "temporal_gpu"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Shared Plotly theme (white and blue enterprise look)
pio.templates["twinmind"] = go.layout.Template(
    layout={
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"color": "#334155", "family": "Segoe UI, Arial, sans-serif", "size": 12},
        "title": {"font": {"color": "#0f172a"}},
        "colorway": ["#2563eb", "#0891b2", "#10b981", "#f59e0b", "#8b5cf6", "#64748b"],
        "xaxis": {"gridcolor": "#e2e8f0", "zerolinecolor": "#cbd5e1", "linecolor": "#cbd5e1"},
        "yaxis": {"gridcolor": "#e2e8f0", "zerolinecolor": "#cbd5e1", "linecolor": "#cbd5e1"},
        "legend": {"bgcolor": "#ffffff"},
        "margin": {"l": 45, "r": 24, "t": 65, "b": 45},
    }
)
pio.templates.default = "twinmind"

MODEL_CANDIDATES = [
    BASE_DIR / "models" / "twinmind_fault_classifier.joblib",
    BASE_DIR / "models" / "motor_fault_pipeline.joblib",
    BASE_DIR / "models" / "model.joblib",
    BASE_DIR / "models" / "pipeline.joblib",
    BASE_DIR / "twinmind_fault_classifier.joblib",
    BASE_DIR / "motor_fault_pipeline.joblib",
    BASE_DIR / "model.joblib",
]

DATA_DIR_CANDIDATES = [
    BASE_DIR / "data" / "data" / "BentlyNevada_System1" / "CombinedFiles",
    BASE_DIR / "data" / "Dataset_digital_twin_predictive_maintenance-main"
    / "data" / "BentlyNevada_System1" / "CombinedFiles",
    BASE_DIR / "Dataset_digital_twin_predictive_maintenance"
    / "data" / "BentlyNevada_System1" / "CombinedFiles",
    BASE_DIR / "data" / "BentlyNevada_System1" / "CombinedFiles",
    BASE_DIR / "data" / "cloud_dataset",
]

GITHUB_DATA_API = (
    "https://api.github.com/repos/"
    "UTK-ASL/Dataset_digital_twin_predictive_maintenance/"
    "contents/data/BentlyNevada_System1/CombinedFiles"
)

FEATURE_COLUMNS_DEFAULT = [
    "ch1_bias", "ch1_derivedPk", "ch1_direct", "ch1_directRMS",
    "ch1_velocityPk", "ch1_velocityRMS",
    "ch2_bias", "ch2_derivedPk", "ch2_direct", "ch2_directRMS",
    "ch2_velocityPk", "ch2_velocityRMS",
]

CLASS_LABELS = {
    "baseline": "Baseline",
    "bent_shaft": "Bent Shaft",
    "eccentric_rotor": "Eccentric Rotor",
    "faulted_bearing": "Faulted Bearing",
    "faulted_coupling": "Faulted Coupling",
    "imbalance": "Imbalance",
    "unknown": "Unknown",
}

# Section names for the top navigation (main dashboard, not the sidebar).
TAB_NAMES = [
    "Overview",
    "Predictions",
    "Sensor Explorer",
    "Anomaly Explorer",
    "Explainability",
    "What-if Simulator",
    "Evaluation",
    "Assistant",
    "AMD GPU",
]

# Operator-facing guidance for each predicted condition (decision support only).
FAULT_GUIDANCE = {
    "baseline": {
        "level": "ok",
        "summary": "Vibration pattern matches normal (baseline) operation.",
        "actions": [
            "Majority of samples are consistent with baseline; review any minority fault predictions before treating this as a healthy reference.",
            "Keep routine monitoring and store this recording as a healthy reference.",
        ],
    },
    "imbalance": {
        "level": "inspect",
        "summary": "Pattern is consistent with rotor imbalance.",
        "actions": [
            "Inspect the rotor for deposits, missing balance weights, or damage.",
            "Plan balancing at the next scheduled stop.",
            "Re-measure vibration after correction.",
        ],
    },
    "bent_shaft": {
        "level": "priority",
        "summary": "Pattern is consistent with a bent shaft.",
        "actions": [
            "Check shaft runout with a dial indicator.",
            "Inspect for recent impact, overload, or thermal distortion.",
            "Escalate to engineering before extended operation.",
        ],
    },
    "eccentric_rotor": {
        "level": "inspect",
        "summary": "Pattern is consistent with an eccentric rotor.",
        "actions": [
            "Check the air gap and rotor seating.",
            "Inspect rotor mounting and bearings for wear.",
            "Compare against baseline vibration after inspection.",
        ],
    },
    "faulted_bearing": {
        "level": "priority",
        "summary": "Pattern is consistent with a bearing fault.",
        "actions": [
            "Inspect the bearing for wear, damage, and lubrication condition.",
            "Check bearing temperature and noise.",
            "Plan replacement before the fault progresses.",
        ],
    },
    "faulted_coupling": {
        "level": "priority",
        "summary": "Pattern is consistent with a coupling fault.",
        "actions": [
            "Inspect the coupling for wear, cracks, or looseness.",
            "Verify shaft alignment.",
            "Check fasteners and the flexible element.",
        ],
    },
    "unknown": {
        "level": "inspect",
        "summary": "The model could not map this recording to a known condition.",
        "actions": ["Have a technician review the recording and the sensors."],
    },
}


# ============================================================
# 2. DATASET DISCOVERY AND DOWNLOAD
# ============================================================

def has_csv_files(directory):
    """Check whether a directory contains CSV recordings."""
    return directory.is_dir() and any(directory.glob("*.csv"))


def find_local_data_directory():
    """Find the dataset already present next to the app."""
    for directory in DATA_DIR_CANDIDATES:
        if has_csv_files(directory):
            return directory

    for root in (BASE_DIR / "data", BASE_DIR):
        if not root.exists():
            continue
        for path in root.rglob("Combined_Baseline1.csv"):
            if has_csv_files(path.parent):
                return path.parent
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def download_public_dataset():
    """Download the seven CSV recordings from the public GitHub repository."""
    cache_dir = Path(tempfile.gettempdir()) / "twinmind_dataset"
    cache_dir.mkdir(parents=True, exist_ok=True)

    allowed_files = {
        "Combined_Baseline1.csv",
        "Combined_Baseline2.csv",
        "Combined_BentShaft.csv",
        "Combined_EccentricRotor.csv",
        "Combined_FaultedBearing.csv",
        "Combined_FaultedCoupling.csv",
        "Combined_Imbalance.csv",
    }

    for cached_file in cache_dir.glob("*.csv"):
        if cached_file.name not in allowed_files:
            cached_file.unlink()

    try:
        response = requests.get(
            GITHUB_DATA_API,
            timeout=30,
            headers={"Accept": "application/vnd.github+json"},
        )
        response.raise_for_status()

        downloaded_count = 0
        for item in response.json():
            if item.get("type") != "file":
                continue
            filename = item.get("name", "")
            if filename not in allowed_files:
                continue

            destination = cache_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                downloaded_count += 1
                continue

            download_url = item.get("download_url")
            if not download_url:
                continue

            file_response = requests.get(download_url, timeout=120)
            file_response.raise_for_status()
            if not file_response.content:
                continue

            destination.write_bytes(file_response.content)
            downloaded_count += 1

        if downloaded_count > 0:
            return str(cache_dir), None
        return None, "The repository API returned no usable combined CSV recordings."

    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def find_data_directory():
    """Prefer the local dataset; otherwise try a public download."""
    local_directory = find_local_data_directory()
    if local_directory is not None:
        return local_directory

    with st.spinner("Looking for the dataset..."):
        downloaded_directory, download_error = download_public_dataset()

    if downloaded_directory:
        return Path(downloaded_directory)

    st.warning(
        "The dataset could not be found locally or downloaded automatically. "
        "You can upload the CSV recordings below."
    )
    if download_error:
        with st.expander("Dataset download details"):
            st.code(download_error)
    return None


# ============================================================
# 3. FILENAME LABEL DETECTION
# ============================================================

def label_from_filename(filename):
    """Infer a class label from long or abbreviated recording names."""
    name = Path(filename).stem.lower()
    normalized = name.replace("-", "_").replace(" ", "_").replace("__", "_")

    if "baseline" in normalized:
        return "baseline"
    if "bentshaft" in normalized or "bent_shaft" in normalized or normalized.endswith("_bs"):
        return "bent_shaft"
    if (
        "eccentricrotor" in normalized
        or "eccentric_rotor" in normalized
        or normalized.endswith("_er")
        or normalized.endswith("_ec")
    ):
        return "eccentric_rotor"
    if "faultedbearing" in normalized or "faulted_bearing" in normalized or normalized.endswith("_fb"):
        return "faulted_bearing"
    if "faultedcoupling" in normalized or "faulted_coupling" in normalized or normalized.endswith("_fc"):
        return "faulted_coupling"
    if "imbalance" in normalized or normalized.endswith("_im"):
        return "imbalance"
    if normalized.endswith("_b1") or normalized.endswith("_b2"):
        return "baseline"
    return "unknown"


def readable_label(label):
    """Format an internal class name for display."""
    key = str(label).lower().strip()
    return CLASS_LABELS.get(key, str(label).replace("_", " ").title())


# ============================================================
# 4. CSV READING
# ============================================================

def clean_column_names(df):
    """Remove extra spaces and byte-order marks from CSV headers."""
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def convert_numeric_columns(df):
    """Convert columns to numeric when their contents allow it."""
    df = df.copy()
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
    return df


@st.cache_data(show_spinner=False)
def read_recording(csv_path, modified_time):
    """Read a CSV file with delimiter fallbacks."""
    del modified_time
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    ]

    best_df, best_score = None, -1
    for options in attempts:
        try:
            candidate = pd.read_csv(csv_path, on_bad_lines="skip", **options)
            candidate = clean_column_names(candidate)
            if len(candidate.columns) > best_score:
                best_df, best_score = candidate, len(candidate.columns)
        except Exception:  # noqa: BLE001
            continue

    if best_df is None or best_df.empty:
        raise ValueError("Could not read a usable CSV table.")
    return convert_numeric_columns(best_df)


@st.cache_data(show_spinner=False)
def load_recordings(directory_string):
    """Load the combined CSV recordings from a directory."""
    directory = Path(directory_string)
    recordings, errors = {}, {}

    for path in sorted(directory.glob("*.csv")):
        if not path.name.lower().startswith("combined"):
            continue
        try:
            df = read_recording(str(path), path.stat().st_mtime)
            if df.empty:
                errors[path.name] = "The CSV contains no usable rows."
                continue

            df = df.copy()
            df["recording_name"] = path.name
            df["actual_label"] = label_from_filename(path.name)
            df["sample_index"] = np.arange(len(df))
            recordings[path.name] = df
        except Exception as exc:  # noqa: BLE001
            errors[path.name] = str(exc)

    return recordings, errors


def read_uploaded_recording(uploaded_file):
    """Read a user-uploaded CSV recording."""
    df = convert_numeric_columns(clean_column_names(pd.read_csv(uploaded_file)))
    df["recording_name"] = uploaded_file.name
    df["actual_label"] = label_from_filename(uploaded_file.name)
    df["sample_index"] = np.arange(len(df))
    return df


# ============================================================
# 5. LOAD SAVED MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path, modified_time):
    """Load the saved model (plain pipeline or a dict containing one)."""
    del modified_time
    saved_object = joblib.load(model_path)

    if isinstance(saved_object, dict):
        pipeline = saved_object.get("pipeline")
        features = saved_object.get("feature_columns", FEATURE_COLUMNS_DEFAULT)
        classes = saved_object.get("classes")
        training_note = saved_object.get("training_note", "")
        if pipeline is None:
            raise ValueError("The saved dictionary does not contain a 'pipeline' entry.")
    else:
        pipeline, features, classes, training_note = (
            saved_object, FEATURE_COLUMNS_DEFAULT, None, "",
        )

    return {
        "pipeline": pipeline,
        "feature_columns": list(features),
        "classes": classes,
        "training_note": training_note,
    }


def find_model_file():
    """Locate the saved classifier."""
    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path
    return None


# ============================================================
# 6. FEATURES AND PREDICTIONS
# ============================================================

def prepare_features(df, feature_columns):
    """Arrange the features in the exact order expected by the model."""
    features = pd.DataFrame(index=df.index)
    for column in feature_columns:
        if column in df.columns:
            features[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            features[column] = np.nan
    return features


def safe_predict(model_info, df):
    """Predict classes and, when supported, probabilities."""
    features = prepare_features(df, model_info["feature_columns"])
    pipeline = model_info["pipeline"]
    predictions = pipeline.predict(features)

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(features)
        except Exception:  # noqa: BLE001
            probabilities = None
    return predictions, probabilities


def add_prediction_columns(df, model_info):
    """Add model predictions to a copy of the recording."""
    result = df.copy()
    predictions, probabilities = safe_predict(model_info, result)

    result["predicted_label"] = [str(v).lower().strip() for v in predictions]
    result["predicted_label_display"] = result["predicted_label"].map(readable_label)

    if probabilities is not None:
        result["prediction_confidence"] = np.max(probabilities, axis=1)
    return result


def find_numeric_columns(df):
    """Return numeric columns excluding metadata."""
    excluded = {
        "sample_index", "recording_name", "actual_label", "predicted_label",
        "predicted_label_display", "prediction_confidence",
    }
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def load_saved_metrics():
    """Load optional previously saved evaluation metrics."""
    metrics_path = BASE_DIR / "models" / "evaluation_metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:  # noqa: BLE001
        return None


# ============================================================
# 7. STYLING, HEADER AND TOP NAVIGATION
# ============================================================

def render_html(markup: str):
    """Strip indentation and blank lines so Streamlit renders HTML, not code."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


inject_theme()

# Hide the sidebar completely: all controls live on the main dashboard.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

render_hero()


# ============================================================
# 8. RESOURCE LOADING
# ============================================================

model_path = find_model_file()
gpu_available, gpu_name, gpu_backend = get_gpu_status()

if model_path is None:
    st.error(
        "Place your saved model at:\n\n`models/twinmind_fault_classifier.joblib`"
    )
    st.stop()

try:
    model_info = load_model(str(model_path), model_path.stat().st_mtime)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model: {exc}")
    st.stop()

data_directory = find_data_directory()

if data_directory is not None:
    try:
        recordings, load_errors = load_recordings(str(data_directory))
    except Exception as exc:  # noqa: BLE001
        recordings, load_errors = {}, {"Dataset": str(exc)}
else:
    recordings, load_errors = {}, {}

if load_errors:
    with st.expander("Dataset loading messages"):
        for filename, message in load_errors.items():
            st.warning(f"{filename}: {message}")

if not recordings:
    st.info(
        "No recordings were loaded. Upload one or more Combined CSV files "
        "below. For accurate class comparisons, use the expected dataset "
        "recordings."
    )
    uploaded_files = st.file_uploader(
        "Upload motor sensor CSV recordings", type=["csv"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded in uploaded_files:
            try:
                recordings[uploaded.name] = read_uploaded_recording(uploaded)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read {uploaded.name}: {exc}")

if not recordings:
    st.stop()


# ============================================================
# 9. RECORDING SELECTION, STATUS AND SUMMARY
# ============================================================

recording_names = list(recordings.keys())

# Open on a faulted recording so the alert is visible immediately.
default_index = next(
    (i for i, n in enumerate(recording_names) if "faultedbearing" in n.lower()),
    next(
        (i for i, n in enumerate(recording_names) if "baseline1" in n.lower()),
        0,
    ),
)

select_col, _spacer_col = st.columns([1, 1])
with select_col:
    selected_recording_name = st.selectbox(
        "Select recording", recording_names, index=default_index
    )

# Top navigation: the nine sections as tabs on the main page.
tabs = st.tabs(TAB_NAMES)

def render_status():
    with st.expander("Project status and model details"):
        status_compute, status_model, status_data = st.columns(3)

        with status_compute:
            st.markdown("**Compute**")
            if gpu_available:
                st.success(f"GPU detected: {gpu_name} ({gpu_backend})")
            else:
                st.info(f"Compute: CPU ({gpu_backend})")

        with status_model:
            st.markdown("**Model**")
            st.success(f"Model found: {model_path.name}")

        with status_data:
            st.markdown("**Dataset**")
            if data_directory is not None:
                st.success("Dataset directory found")
                st.caption(f"Dataset: {data_directory.name}")
            else:
                st.info("Using uploaded recordings")

        st.caption(
            "The fault classifier is a scikit-learn model and runs on the CPU. "
            "The GPU is used by the live screening in the AMD GPU section. The "
            "Assistant section reads the saved AMD screening reports."
        )

        st.markdown("**Required model features**")
        st.markdown("  ".join(f"`{f}`" for f in model_info["feature_columns"]))

        if data_directory is not None:
            st.markdown("**Dataset location**")
            st.code(str(data_directory), language=None)

raw_df = recordings[selected_recording_name].copy()

try:
    selected_df = add_prediction_columns(raw_df, model_info)
except Exception as exc:  # noqa: BLE001
    st.error(f"Prediction failed: {exc}")
    st.write("Required model features:", model_info["feature_columns"])
    st.write("CSV columns:", list(raw_df.columns))
    st.stop()

numeric_columns = find_numeric_columns(selected_df)
model_features = model_info["feature_columns"]

actual_label = label_from_filename(selected_recording_name)
prediction_counts = selected_df["predicted_label"].value_counts()
most_frequent_prediction = (
    str(prediction_counts.index[0]) if not prediction_counts.empty else "unknown"
)

missing_feature_count = sum(
    1
    for feature in model_features
    if feature not in selected_df.columns
    or pd.to_numeric(selected_df[feature], errors="coerce").notna().sum() == 0
)

# Row-level input quality checks (predictions are not changed by this check).
feature_frame = prepare_features(selected_df, model_features)
missing_cells_per_feature = feature_frame.isna().sum()
missing_sensor_cells = int(missing_cells_per_feature.sum())
total_sensor_cells = int(feature_frame.size)
missing_sensor_percent = (
    100.0 * missing_sensor_cells / total_sensor_cells if total_sensor_cells else 0.0
)

def render_quality():
    if missing_sensor_cells:
        st.warning(
            f"Input quality warning: {missing_sensor_cells:,} missing or "
            f"non-numeric sensor values ({missing_sensor_percent:.2f}% of "
            "required sensor values). The model pipeline may impute these "
            "values, so predictions may be less reliable."
        )
    else:
        st.success(
            "Input quality check: all required sensor values are present and numeric."
        )

# Distribution-shift screening against the other recordings (heuristic only).
# The calculation always runs because the review flag below depends on it.
shifted_features = []
shift_results = []
reference_frames = []

for reference_name, reference_df in recordings.items():
    if reference_name == selected_recording_name:
        continue
    if "QualityTest" in str(reference_name):
        continue
    reference_frames.append(prepare_features(reference_df, model_features))

if reference_frames:
    reference_data = pd.concat(reference_frames, axis=0, ignore_index=True)
    selected_features = prepare_features(raw_df, model_features)

    lower_bounds = reference_data.quantile(0.01)
    upper_bounds = reference_data.quantile(0.99)

    for feature in model_features:
        reference_values = reference_data[feature].dropna()
        current_values = selected_features[feature].dropna()
        if reference_values.empty or current_values.empty:
            continue

        outside = (current_values < lower_bounds[feature]) | (
            current_values > upper_bounds[feature]
        )
        shift_results.append((feature, 100.0 * float(outside.mean())))

    shifted_features = [(f, p) for f, p in shift_results if p > 20.0]
    shifted_features.sort(key=lambda item: item[1], reverse=True)

def render_shift():
    if reference_frames:
        st.subheader("Distribution-shift screening")
        st.caption(
            "Compares this recording with other available recordings. "
            "The 1st-99th percentile bounds and 20% trigger are heuristic "
            "screening settings, not validated operating limits."
        )

        shift_table = pd.DataFrame(
            shift_results, columns=["Sensor feature", "Outside reference range (%)"]
        )
        if not shift_table.empty:
            shift_table["Outside reference range (%)"] = shift_table[
                "Outside reference range (%)"
            ].round(2)
            shift_table["Screening result"] = shift_table[
                "Outside reference range (%)"
            ].apply(lambda v: "Review" if v > 20.0 else "Below threshold")
            shift_table = shift_table.sort_values(
                "Outside reference range (%)", ascending=False
            )
            st.markdown("**Sensor shift details**")
            st.dataframe(shift_table, width="stretch", hide_index=True)

        if shifted_features:
            details = "; ".join(
                f"{f}: {p:.1f}% outside reference range" for f, p in shifted_features[:5]
            )
            st.warning(
                "Potential sensor distribution shift detected. "
                + details
                + ". Review sensor scaling, operating conditions, and data "
                "quality before interpreting model predictions."
            )
        elif shift_results:
            st.info(
                "No strong distribution shift was flagged by this screening "
                "check. This does not establish that the model is reliable "
                "for this recording."
            )
        else:
            st.info(
                "Distribution shift could not be assessed because there were "
                "not enough valid reference or selected sensor values."
            )
    else:
        st.info(
            "Distribution shift could not be assessed because no other "
            "reference recordings are available."
        )

# ------------------------------------------------------------
# Export real classifier results for the Evolus assistant
# ------------------------------------------------------------

def render_export():
    st.subheader("Export for Evolus")
    st.caption(
        "Export real predictions and sensor summaries for the TwinMind "
        "Industrial Assistant. This is decision support, not a confirmed "
        "physical diagnosis."
    )

    def _export_number(value):
        return None if pd.isna(value) else float(value)

    export_features = [f for f in model_features if f in selected_df.columns]
    export_columns = [
        c
        for c in ["sample_index", *export_features, "predicted_label", "prediction_confidence"]
        if c in selected_df.columns
    ]

    feature_summary = {}
    for feature in export_features:
        values = pd.to_numeric(selected_df[feature], errors="coerce")
        valid = values.dropna()
        feature_summary[feature] = {
            "min": _export_number(valid.min()) if not valid.empty else None,
            "median": _export_number(valid.median()) if not valid.empty else None,
            "max": _export_number(valid.max()) if not valid.empty else None,
            "missing_or_non_numeric": int(values.isna().sum()),
        }

    prediction_distribution = {
        str(label): int(count)
        for label, count in selected_df["predicted_label"].value_counts(dropna=False).items()
    }

    confidence_summary = None
    if "prediction_confidence" in selected_df.columns:
        conf_values = pd.to_numeric(
            selected_df["prediction_confidence"], errors="coerce"
        ).dropna()
        if not conf_values.empty:
            confidence_summary = {
                "mean_model_score": float(conf_values.mean()),
                "note": "Model score, not a guarantee of correctness or a calibrated probability.",
            }

    sample_json = selected_df[export_columns].head(5).to_json(orient="records")

    evolus_report = {
        "project": "TwinMind AI",
        "recording": str(selected_recording_name),
        "samples_analyzed": int(len(selected_df)),
        "model_features": export_features,
        "most_frequent_prediction": str(most_frequent_prediction),
        "prediction_distribution": prediction_distribution,
        "filename_label_inferred_not_verified": str(actual_label),
        "input_quality": {
            "missing_or_non_numeric_cells": int(missing_sensor_cells),
            "total_required_sensor_cells": int(total_sensor_cells),
            "missing_percent": round(float(missing_sensor_percent), 3),
            "missing_by_feature": {
                str(f): int(c) for f, c in missing_cells_per_feature.items()
            },
        },
        "sensor_feature_summary": feature_summary,
        "confidence_summary": confidence_summary,
        "sample_predictions_first_5_rows": json.loads(sample_json),
        "limitations": [
            "Historical recording; not live machine monitoring.",
            "Classifier predictions are estimates, not confirmed faults.",
            "Filename label is inferred from the filename, not independently verified.",
            "Feature summaries describe this recording, not manufacturer safety limits.",
            "Only the first five sample predictions are included in this export.",
        ],
    }

    st.download_button(
        label="Download Evolus report (JSON)",
        data=json.dumps(evolus_report, indent=2, ensure_ascii=False),
        file_name="twinmind_evolus_report.json",
        mime="application/json",
        key="download_evolus_report",
    )


def render_glance():
    st.subheader("Recording at a glance")
    st.caption(
        "Summary for the selected historical recording. Predictions are model "
        "estimates, not confirmed diagnoses of a physical motor."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensor samples", f"{len(selected_df):,}")
    m2.metric("Model features", len(model_features))
    m3.metric("Missing feature columns", missing_feature_count)
    m4.metric("Most frequent prediction", readable_label(most_frequent_prediction))

    if actual_label != "unknown":
        if most_frequent_prediction == actual_label:
            st.success(
                "The most frequent prediction agrees with the label inferred "
                "from this recording's filename."
            )
        else:
            st.warning(
                "The most frequent prediction differs from the filename-based "
                "label. Investigate the data and model before drawing conclusions."
            )

    st.caption(
        f"Recording: `{selected_recording_name}` · "
        f"Filename-based label: **{readable_label(actual_label)}**"
    )


# ============================================================
# 10. DIAGNOSIS AND RECOMMENDED ACTION
# ============================================================

def style_chart(fig, height=None):
    """Apply a consistent light enterprise style to Plotly charts."""
    options = dict(
        template="twinmind",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#334155"),
        margin=dict(l=24, r=24, t=65, b=30),
        title=dict(font=dict(size=17, color="#0f172a"), x=0.02, xanchor="left"),
        legend=dict(bgcolor="#ffffff", font=dict(color="#475569", size=11)),
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor="#cbd5e1", font=dict(color="#0f172a")
        ),
    )
    if height is not None:
        options["height"] = height
    fig.update_layout(**options)
    axis = dict(
        showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1", tickfont=dict(color="#475569"),
        title_font=dict(color="#334155"),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


guide = FAULT_GUIDANCE.get(most_frequent_prediction, FAULT_GUIDANCE["unknown"])
top_share = (
    100.0 * float(prediction_counts.iloc[0]) / len(selected_df)
    if not prediction_counts.empty and len(selected_df)
    else 0.0
)
mean_conf = None
if "prediction_confidence" in selected_df.columns:
    _mc = selected_df["prediction_confidence"].mean()
    mean_conf = float(_mc) if pd.notna(_mc) else None

# Heuristic review triggers, not validated limits.
needs_review = (
    top_share < 70.0
    or (mean_conf is not None and mean_conf < 0.70)
    or missing_sensor_cells > 0
    or bool(shifted_features)
)

def render_diagnosis():
    st.subheader("Diagnosis and recommended action")
    d1, d2, d3 = st.columns(3)
    d1.metric("Detected condition", readable_label(most_frequent_prediction))
    d2.metric("Samples agreeing", f"{top_share:.1f}%")
    d3.metric("Mean model score", f"{mean_conf * 100:.1f}%" if mean_conf is not None else "n/a")

    if guide["level"] == "ok":
        st.success(guide["summary"])
    else:
        st.error(f"ALERT: {guide['summary']}")

    if needs_review:
        st.warning(
            "Human review recommended: low sample agreement, low model score, "
            "missing sensor values, or a distribution-shift flag was raised."
        )

    st.markdown("**Recommended actions**")
    for action in guide["actions"]:
        st.markdown(f"- {action}")
    st.caption(
        "Decision support only. A qualified technician must verify any "
        "suspected fault before maintenance decisions are made."
    )


# ============================================================
# 11. DASHBOARD SECTIONS (tabs on the main page)
# ============================================================

# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------

with tabs[0]:
    render_glance()
    render_diagnosis()
    st.subheader("Recording overview")
    left, right = st.columns([1.5, 1])

    with left:
        if numeric_columns:
            overview_column = st.selectbox(
                "Select a numeric column", numeric_columns, key="overview_column"
            )
            chart_df = selected_df[["sample_index", overview_column]].dropna()
            fig = px.line(
                chart_df, x="sample_index", y=overview_column,
                title=f"{overview_column} across recorded samples",
            )
            style_chart(fig, height=400)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numeric columns were detected.")

    with right:
        distribution = (
            selected_df["predicted_label"].value_counts()
            .rename_axis("Class").reset_index(name="Samples")
        )
        distribution["Class"] = distribution["Class"].map(readable_label)
        fig = px.pie(
            distribution, names="Class", values="Samples", hole=0.45,
            title="Prediction distribution",
        )
        style_chart(fig, height=400)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Recording details")
    st.dataframe(
        pd.DataFrame(
            {
                "Property": [
                    "Recording file", "Filename-based label", "Number of samples",
                    "Numeric columns", "Required model features",
                ],
                "Value": [
                    str(selected_recording_name), str(readable_label(actual_label)),
                    str(len(selected_df)), str(len(numeric_columns)),
                    str(len(model_features)),
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    with st.expander("Preview data"):
        st.dataframe(selected_df.head(30), hide_index=True, width="stretch")

    with st.expander("Data quality, distribution shift and Evolus export"):
        render_quality()
        render_shift()
        render_export()

    render_status()

# ------------------------------------------------------------
# TAB 2: PREDICTIONS
# ------------------------------------------------------------

with tabs[1]:
    st.subheader("Motor-condition predictions")
    st.write(
        "Predictions come from the saved classifier. Confidence values, "
        "when available, are model scores and are not guaranteed to "
        "represent calibrated probabilities."
    )

    prediction_summary = (
        selected_df["predicted_label"].value_counts()
        .rename_axis("Class").reset_index(name="Sample count")
    )
    prediction_summary["Class"] = prediction_summary["Class"].map(readable_label)
    prediction_summary["Percentage"] = (
        prediction_summary["Sample count"] / len(selected_df) * 100
    ).round(2)
    st.dataframe(prediction_summary, hide_index=True, width="stretch")

    if "prediction_confidence" in selected_df.columns:
        confidence = selected_df["prediction_confidence"].dropna()
        if not confidence.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean confidence", f"{confidence.mean() * 100:.1f}%")
            c2.metric("Minimum confidence", f"{confidence.min() * 100:.1f}%")
            c3.metric("Maximum confidence", f"{confidence.max() * 100:.1f}%")

            fig = px.histogram(
                selected_df.dropna(subset=["prediction_confidence"]),
                x="prediction_confidence", nbins=30,
                title="Model confidence distribution",
            )
            style_chart(fig)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Individual sample predictions")
    row_limit = st.slider(
        "Rows to display",
        min_value=1,
        max_value=len(selected_df),
        value=min(100, len(selected_df)),
    )
    display_columns = [
        c
        for c in ["sample_index", "actual_label", "predicted_label_display", "prediction_confidence"]
        if c in selected_df.columns
    ]
    st.dataframe(selected_df[display_columns].head(row_limit), hide_index=True, width="stretch")

    st.download_button(
        "Download predictions CSV",
        data=selected_df.to_csv(index=False).encode("utf-8"),
        file_name="twinmind_predictions.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------
# TAB 3: SENSOR EXPLORER
# ------------------------------------------------------------

with tabs[2]:
    st.subheader("Sensor Explorer")

    available_sensors = [
        f for f in model_features
        if f in selected_df.columns and pd.api.types.is_numeric_dtype(selected_df[f])
    ]

    if not available_sensors:
        st.warning("The required sensor features were not found in this recording.")
    else:
        chosen_sensors = st.multiselect(
            "Select sensor features",
            available_sensors,
            default=available_sensors[: min(3, len(available_sensors))],
        )

        if chosen_sensors:
            sensor_layout = st.radio(
                "Chart layout",
                SENSOR_LAYOUTS,
                horizontal=True,
                key="sensor_layout",
                help="Sensors have very different scales (for example ch1_bias "
                "is near -12). Separate panels or z-score normalization keep "
                "every signal readable.",
            )
            fig, chart_height = sensor_figure(
                selected_df, chosen_sensors, sensor_layout
            )
            style_chart(fig, height=chart_height)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Sensor statistics")
        stats = selected_df[available_sensors].describe().T
        stats.index.name = "Sensor"
        st.dataframe(stats, width="stretch")

        if len(available_sensors) >= 2:
            st.subheader("Sensor correlation")
            correlation = selected_df[available_sensors].corr(numeric_only=True)
            fig = px.imshow(
                correlation, text_auto=".2f", aspect="auto",
                title="Correlation between sensor features",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            )
            style_chart(fig)
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# TAB 4: ANOMALY EXPLORER
# ------------------------------------------------------------

with tabs[3]:
    st.subheader("Anomaly Explorer")
    st.write(
        "This screen flags unusual values using rolling median absolute "
        "deviation. It is an exploratory statistical method, not a validated "
        "industrial alarm or fault diagnosis."
    )

    anomaly_sensors = [
        f for f in model_features
        if f in selected_df.columns
        and pd.to_numeric(selected_df[f], errors="coerce").notna().any()
    ]

    if not anomaly_sensors:
        st.warning("No usable model sensor features are available.")
    else:
        anomaly_sensor = st.selectbox("Sensor to inspect", anomaly_sensors, key="anomaly_sensor")
        window = st.slider("Rolling window", min_value=15, max_value=301, value=51, step=2)
        threshold = st.slider(
            "Robust deviation threshold", min_value=2.0, max_value=8.0, value=3.5, step=0.5
        )

        signal = pd.to_numeric(selected_df[anomaly_sensor], errors="coerce")
        min_periods = max(3, window // 5)

        rolling_median = signal.rolling(window=window, center=True, min_periods=min_periods).median()
        rolling_mad = (signal - rolling_median).abs().rolling(
            window=window, center=True, min_periods=min_periods
        ).median()

        global_mad = (signal - signal.median()).abs().median()
        global_scale = 1.4826 * global_mad
        if not np.isfinite(global_scale) or global_scale == 0:
            global_scale = signal.std()
        if not np.isfinite(global_scale) or global_scale == 0:
            global_scale = 1.0

        robust_scale = (1.4826 * rolling_mad).replace(0, np.nan).fillna(global_scale)
        score = (signal - rolling_median).abs() / robust_scale
        anomaly_mask = (score > threshold).fillna(False)

        anomaly_count = int(anomaly_mask.sum())
        anomaly_percentage = anomaly_count / max(1, signal.notna().sum()) * 100

        a1, a2, a3 = st.columns(3)
        a1.metric("Flagged samples", f"{anomaly_count:,}")
        a2.metric("Flagged percentage", f"{anomaly_percentage:.2f}%")
        a3.metric("Selected sensor", anomaly_sensor)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=selected_df["sample_index"], y=signal, mode="lines", name="Sensor reading"))
        fig.add_trace(go.Scatter(x=selected_df["sample_index"], y=rolling_median, mode="lines", name="Rolling median"))
        flagged_rows = selected_df.loc[anomaly_mask]
        fig.add_trace(
            go.Scatter(
                x=flagged_rows["sample_index"], y=signal.loc[anomaly_mask],
                mode="markers", name="Flagged sample",
                marker={"symbol": "x", "size": 8},
            )
        )
        fig.update_layout(
            title=f"Potential anomalies: {anomaly_sensor}",
            xaxis_title="Sample index", yaxis_title="Sensor reading",
        )
        style_chart(fig, height=480)
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Unusual readings can have several causes, including sensor "
            "issues, operating changes, or real mechanical changes. "
            "The cause must be investigated separately."
        )

        if anomaly_count:
            anomaly_columns = [
                c for c in ["sample_index", anomaly_sensor, "predicted_label_display"]
                if c in selected_df.columns
            ]
            st.dataframe(
                selected_df.loc[anomaly_mask, anomaly_columns].head(500),
                hide_index=True, width="stretch",
            )
        else:
            st.success("No samples exceeded the selected threshold.")

# ------------------------------------------------------------
# TAB 5: EXPLAINABILITY
# ------------------------------------------------------------

with tabs[4]:
    st.subheader("Model Explainability")
    st.write(
        "Feature importance shows how a supported model uses input features. "
        "It does not establish causation or prove a mechanical fault."
    )

    pipeline = model_info["pipeline"]
    estimator = pipeline
    if hasattr(pipeline, "named_steps"):
        for step in reversed(list(pipeline.named_steps.values())):
            if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                estimator = step
                break

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        if len(values) == len(model_features):
            importance_df = pd.DataFrame(
                {"Feature": model_features, "Importance": values}
            ).sort_values("Importance", ascending=False)
            fig = px.bar(
                importance_df, x="Importance", y="Feature", orientation="h",
                title="Feature importance",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            style_chart(fig, height=480)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(importance_df, hide_index=True, width="stretch")
        else:
            st.info("Feature importance could not be aligned with the saved feature names.")

    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        values = np.abs(coefficients) if coefficients.ndim == 1 else np.mean(np.abs(coefficients), axis=0)
        if len(values) == len(model_features):
            importance_df = pd.DataFrame(
                {"Feature": model_features, "Absolute coefficient": values}
            ).sort_values("Absolute coefficient", ascending=False)
            fig = px.bar(
                importance_df, x="Absolute coefficient", y="Feature", orientation="h",
                title="Absolute model coefficients",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            style_chart(fig, height=480)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("The coefficient count does not match the feature list.")

    else:
        st.info(
            "This model does not expose feature_importances_ or coef_. "
            "Feature importance is unavailable through this interface."
        )

    if model_info.get("training_note"):
        st.subheader("Model training note")
        st.write(model_info["training_note"])

# ------------------------------------------------------------
# TAB 6: WHAT-IF SIMULATOR
# ------------------------------------------------------------

with tabs[5]:
    st.subheader("What-if Simulator")
    st.write(
        "Change sensor values to see how the classifier responds to a "
        "hypothetical sample. These are simulated inputs, not live readings."
    )

    available_features = [
        f for f in model_features
        if f in selected_df.columns
        and pd.to_numeric(selected_df[f], errors="coerce").notna().any()
    ]

    if len(available_features) != len(model_features):
        st.warning(
            "Some required features are missing from this recording. "
            "The simulator is disabled until all model features are available."
        )
    else:
        source_index = st.slider(
            "Choose a source sample",
            min_value=0,
            max_value=len(selected_df) - 1,
            value=min(100, len(selected_df) - 1),
        )
        source_row = selected_df.iloc[source_index]
        what_if = {}
        slider_columns = st.columns(2)

        for index, feature in enumerate(model_features):
            values = pd.to_numeric(selected_df[feature], errors="coerce").dropna()
            if values.empty:
                continue

            current_value = float(source_row[feature])
            if not np.isfinite(current_value):
                current_value = float(values.median())

            minimum = min(float(values.quantile(0.01)), current_value)
            maximum = max(float(values.quantile(0.99)), current_value)

            if minimum == maximum:
                padding = max(abs(minimum) * 0.1, 0.001)
                minimum -= padding
                maximum += padding

            step = (maximum - minimum) / 200
            if not np.isfinite(step) or step <= 0:
                step = 0.001

            with slider_columns[index % 2]:
                what_if[feature] = st.slider(
                    feature,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(current_value),
                    step=float(step),
                    key=f"whatif_{feature}",
                )

        if st.button("Run what-if prediction", type="primary"):
            input_df = pd.DataFrame([{f: what_if[f] for f in model_features}])
            try:
                prediction = model_info["pipeline"].predict(input_df)[0]
                st.success(
                    f"Hypothetical predicted class: **{readable_label(prediction)}**"
                )

                if hasattr(model_info["pipeline"], "predict_proba"):
                    try:
                        probabilities = model_info["pipeline"].predict_proba(input_df)[0]
                        classes = model_info["pipeline"].classes_
                        probability_df = pd.DataFrame(
                            {
                                "Class": [readable_label(v) for v in classes],
                                "Probability": probabilities,
                            }
                        ).sort_values("Probability", ascending=False)
                        fig = px.bar(
                            probability_df, x="Class", y="Probability",
                            title="Hypothetical class probabilities",
                        )
                        fig.update_layout(yaxis_tickformat=".0%")
                        style_chart(fig)
                        st.plotly_chart(fig, width="stretch")
                    except Exception as exc:  # noqa: BLE001
                        st.info(
                            "The prediction succeeded, but the probability "
                            f"chart could not be displayed: {exc}"
                        )

                st.warning(
                    "This is only the classifier's response to hypothetical "
                    "values. It does not confirm the condition of a real motor."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"What-if prediction failed: {exc}")

# ------------------------------------------------------------
# TAB 7: EVALUATION
# ------------------------------------------------------------

with tabs[6]:
    st.subheader("Model Evaluation")
    render_temporal_evaluation(TEMPORAL_DIR)
    st.divider()
    st.subheader("Per-recording agreement check")
    st.write(
        "The reference label below is inferred from the recording filename. "
        "It is not an independent, sample-by-sample engineering diagnosis. "
        "A single recording contains one class, so high agreement here is "
        "expected and is not a generalization result."
    )

    labelled_df = selected_df[selected_df["actual_label"] != "unknown"].copy()

    if labelled_df.empty:
        st.info("No known filename-based labels are available.")
    else:
        y_true = labelled_df["actual_label"].astype(str)
        y_pred = labelled_df["predicted_label"].astype(str)
        agreement = accuracy_score(y_true, y_pred)

        e1, e2, e3 = st.columns(3)
        e1.metric("Row-level agreement", f"{agreement * 100:.2f}%")
        e2.metric("Evaluated samples", f"{len(labelled_df):,}")
        e3.metric("Reference label", readable_label(actual_label))

        labels = sorted(set(y_true) | set(y_pred))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        matrix_df = pd.DataFrame(
            matrix,
            index=[readable_label(l) for l in labels],
            columns=[readable_label(l) for l in labels],
        )

        fig = px.imshow(
            matrix_df, text_auto=True, aspect="auto",
            title="Confusion matrix for selected recording",
            labels={
                "x": "Predicted class",
                "y": "Filename-based reference class",
                "color": "Samples",
            },
        )
        style_chart(fig)
        st.plotly_chart(fig, width="stretch")

        report = classification_report(
            y_true, y_pred, labels=labels,
            target_names=[readable_label(l) for l in labels],
            output_dict=True, zero_division=0,
        )
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    saved_metrics = load_saved_metrics()
    if saved_metrics is not None:
        with st.expander("Saved evaluation metrics"):
            st.json(saved_metrics)

    st.warning(
        "Evaluation limitation: rows from one recording may be correlated. "
        "Agreement on a recording does not demonstrate generalization to "
        "a new motor or independent test run. Validate on independent "
        "recordings before considering operational use."
    )

# ------------------------------------------------------------
# TAB 8: MAINTENANCE ASSISTANT
# ------------------------------------------------------------

with tabs[7]:
    render_maintenance_assistant(
        selected_recording_name,
        gpu_available,
        gpu_name,
        gpu_backend,
    )

# ------------------------------------------------------------
# TAB 9: AMD GPU EVIDENCE
# ------------------------------------------------------------

with tabs[8]:
    render_live_gpu_screening(selected_df, model_features, selected_recording_name)
    st.divider()
    st.subheader("AMD GPU Analysis")
    st.caption(
        "Previously computed AMD Radeon Pro W7900D workload using PyTorch and "
        "ROCm. These are saved results, not live processing of the currently "
        "selected recording."
    )

    screening_path = AMD_DIR / "twinmind_gpu_sensor_screening.csv"
    summary_path = AMD_DIR / "twinmind_gpu_feature_summary.csv"

    try:
        amd_results = pd.read_csv(screening_path)
        amd_features = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        st.error(f"Could not load AMD result files: {exc}")
    else:
        required_results = {"source_file", "source_row", "screening_score", "flagged_for_review"}
        required_features = {"feature", "mean_after_imputation", "standard_deviation"}

        if not required_results.issubset(amd_results.columns):
            st.error("The AMD screening CSV has an unexpected structure.")
        elif not required_features.issubset(amd_features.columns):
            st.error("The AMD feature summary CSV has an unexpected structure.")
        else:
            flagged_mask = (
                amd_results["flagged_for_review"].astype(str).str.lower().isin(["true", "1", "yes"])
            )
            flagged_count = int(flagged_mask.sum())

            st.info(
                "These statistical screening scores identify samples for "
                "review; they do not confirm physical machine faults."
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Samples screened", f"{len(amd_results):,}")
            col2.metric("Features processed", f"{len(amd_features):,}")
            col3.metric("Flagged for review", f"{flagged_count:,}")

            render_gpu_session_details(AMD_DIR)

            # ---------------- Maintenance action workflow ----------------
            st.markdown("**Maintenance Action Workflow**")
            st.caption(
                "Choose a flagged sample, review the evidence, record an "
                "inspection outcome, and save the review log."
            )

            if flagged_count == 0:
                st.success("No samples are currently flagged for review.")
            else:
                candidates = (
                    amd_results.loc[flagged_mask]
                    .sort_values("screening_score", ascending=False)
                    .head(200)
                )

                sample_labels = {}
                for sample_index, sample in candidates.iterrows():
                    label = (
                        f"{sample['source_file']} | row {sample['source_row']} | "
                        f"score {float(sample['screening_score']):.4f}"
                    )
                    sample_labels[label] = sample_index

                selected_label = st.selectbox(
                    "Select a flagged sensor sample",
                    options=list(sample_labels.keys()),
                    key="maintenance_selected_sample",
                )
                selected_sample = candidates.loc[sample_labels[selected_label]]

                st.warning(
                    "ALERT: This sample was flagged for further review. "
                    "This is a statistical alert, not a confirmed failure."
                )

                info_col1, info_col2 = st.columns(2)
                info_col1.metric("Screening score", f"{float(selected_sample['screening_score']):.4f}")
                info_col2.metric("Source row", str(selected_sample["source_row"]))

                st.markdown("**Why was this sample flagged?**")
                st.write(
                    "Its statistical screening score exceeded the configured "
                    "screening threshold. The score prioritizes samples for "
                    "investigation; it does not identify a specific failed "
                    "component by itself."
                )
                st.write(f"Source dataset: {selected_sample['source_file']}")

                st.markdown("**Recommended inspection steps**")
                st.markdown(
                    "- Open the original dataset and inspect the selected row.\n"
                    "- Compare its sensor readings with normal operating "
                    "conditions and nearby samples.\n"
                    "- Check vibration readings, bearing condition, shaft "
                    "alignment, and coupling condition as applicable.\n"
                    "- Follow site safety procedures and have a qualified "
                    "technician verify any suspected issue.\n"
                    "- Record the outcome before deciding on maintenance."
                )

                review_status = st.selectbox(
                    "Inspection outcome",
                    [
                        "Inspection pending",
                        "Inspected — no issue found",
                        "Maintenance required",
                        "Escalated for engineering review",
                    ],
                    key="maintenance_review_status",
                )
                review_notes = st.text_area(
                    "Technician notes",
                    placeholder="Record observations, checks performed, and any follow-up required...",
                    key="maintenance_review_notes",
                )
                st.caption(
                    "Only save an outcome that reflects the actual review. "
                    "The log is stored locally in this project."
                )

                if st.button("Save maintenance review", key="save_maintenance_review", type="primary"):
                    log_path = AMD_DIR / "maintenance_review_log.csv"
                    log_path.parent.mkdir(parents=True, exist_ok=True)

                    record = pd.DataFrame(
                        [
                            {
                                "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "source_file": selected_sample["source_file"],
                                "source_row": selected_sample["source_row"],
                                "screening_score": selected_sample["screening_score"],
                                "review_status": review_status,
                                "technician_notes": review_notes.strip(),
                            }
                        ]
                    )
                    if log_path.exists():
                        record = pd.concat([pd.read_csv(log_path), record], ignore_index=True)
                    record.to_csv(log_path, index=False)
                    st.success(f"Review saved to {log_path.name}")

            # ---------------- Score distribution and tables ----------------
            st.markdown("**Screening score distribution**")
            plot_df = amd_results.copy()
            plot_df["Review status"] = flagged_mask.map(
                {True: "Flagged for review", False: "Below threshold"}
            )
            score_fig = px.histogram(
                plot_df, x="screening_score", color="Review status", nbins=40,
                title="Distribution of statistical screening scores",
                labels={"screening_score": "Screening score"},
            )
            style_chart(score_fig, height=380)
            st.plotly_chart(score_fig, width="stretch")

            st.markdown("**Samples flagged for review**")
            show_flagged_only = st.checkbox(
                "Show flagged samples only", value=True, key="amd_show_flagged_only"
            )
            display_df = amd_results.loc[flagged_mask] if show_flagged_only else amd_results
            display_df = display_df.sort_values("screening_score", ascending=False)

            st.dataframe(display_df.head(200), width="stretch", hide_index=True)
            st.caption(
                "At most 200 rows are shown. Source row numbers refer to the "
                "saved screening output. Inspect the original recording before "
                "interpreting individual samples."
            )

            st.markdown("**Feature statistics after imputation**")
            feature_fig = px.bar(
                amd_features.sort_values("standard_deviation", ascending=False),
                x="feature", y="standard_deviation",
                title="Standard deviation by sensor feature",
                labels={"feature": "Sensor feature", "standard_deviation": "Standard deviation"},
            )
            style_chart(feature_fig, height=400)
            st.plotly_chart(feature_fig, width="stretch")
            st.dataframe(amd_features, width="stretch", hide_index=True)


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()
st.caption(
    "TwinMind AI · Research and demonstration prototype · "
    "Historical sensor recordings only · Not a safety-critical monitoring system"
)