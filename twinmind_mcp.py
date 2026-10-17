"""TwinMind MCP server: exposes the fault classifier, recordings and guidance as agent tools."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
mcp = FastMCP("twinmind")

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
FEATURES_DEFAULT = [
    "ch1_bias", "ch1_derivedPk", "ch1_direct", "ch1_directRMS", "ch1_velocityPk", "ch1_velocityRMS",
    "ch2_bias", "ch2_derivedPk", "ch2_direct", "ch2_directRMS", "ch2_velocityPk", "ch2_velocityRMS",
]

GUIDANCE = {
    "baseline": {"level": "ok", "summary": "Vibration pattern matches normal (baseline) operation.",
                 "actions": ["Keep routine monitoring.", "Store this recording as a healthy reference."]},
    "imbalance": {"level": "inspect", "summary": "Consistent with rotor imbalance.",
                  "actions": ["Inspect the rotor for deposits, missing balance weights or damage.",
                              "Plan balancing at the next scheduled stop.", "Re-measure after correction."]},
    "bent_shaft": {"level": "priority", "summary": "Consistent with a bent shaft.",
                   "actions": ["Check shaft runout with a dial indicator.",
                               "Inspect for impact, overload or thermal distortion.",
                               "Escalate to engineering before extended operation."]},
    "eccentric_rotor": {"level": "inspect", "summary": "Consistent with an eccentric rotor.",
                        "actions": ["Check the air gap and rotor seating.",
                                    "Inspect rotor mounting and bearings for wear.",
                                    "Compare against baseline vibration after inspection."]},
    "faulted_bearing": {"level": "priority", "summary": "Consistent with a bearing fault.",
                        "actions": ["Inspect the bearing for wear, damage and lubrication.",
                                    "Check bearing temperature and noise.",
                                    "Plan replacement before the fault progresses."]},
    "faulted_coupling": {"level": "priority", "summary": "Consistent with a coupling fault.",
                         "actions": ["Inspect the coupling for wear, cracks or looseness.",
                                     "Verify shaft alignment.", "Check fasteners and the flexible element."]},
    "unknown": {"level": "inspect", "summary": "Could not map this to a known condition.",
                "actions": ["Have a technician review the recording and the sensors."]},
}
DISCLAIMER = "Decision support only; a qualified technician must verify any suspected fault."

_state: dict = {}


def _label_from_name(name: str) -> str:
    n = Path(name).stem.lower().replace("-", "_").replace(" ", "_")
    for key, label in [("baseline", "baseline"), ("bentshaft", "bent_shaft"), ("bent_shaft", "bent_shaft"),
                       ("eccentric", "eccentric_rotor"), ("faultedbearing", "faulted_bearing"),
                       ("faulted_bearing", "faulted_bearing"), ("faultedcoupling", "faulted_coupling"),
                       ("faulted_coupling", "faulted_coupling"), ("imbalance", "imbalance")]:
        if key in n:
            return label
    return "unknown"


def _model():
    if "model" not in _state:
        path = next((p for p in MODEL_CANDIDATES if p.is_file()), None)
        if path is None:
            raise FileNotFoundError("Model file not found in models/ or the project root.")
        obj = joblib.load(path)
        if isinstance(obj, dict):
            pipe = obj.get("pipeline")
            feats = list(obj.get("feature_columns", FEATURES_DEFAULT))
        else:
            pipe, feats = obj, FEATURES_DEFAULT
        _state["model"] = (pipe, feats)
    return _state["model"]


def _data_dir():
    if "dir" not in _state:
        d = next((p for p in DATA_DIR_CANDIDATES if p.is_dir() and any(p.glob("*.csv"))), None)
        if d is None:
            for root in (BASE_DIR / "data", BASE_DIR):
                hit = next(root.rglob("Combined_Baseline1.csv"), None) if root.exists() else None
                if hit:
                    d = hit.parent
                    break
        if d is None:
            raise FileNotFoundError("Dataset folder with Combined_*.csv not found.")
        _state["dir"] = d
    return _state["dir"]


def _recordings() -> dict:
    return {p.name: p for p in sorted(_data_dir().glob("Combined*.csv"))}


def _resolve(name: str) -> Path:
    files = _recordings()
    q = name.lower().replace(".csv", "").replace("combined_", "").replace(" ", "").replace("_", "")
    for fname, path in files.items():
        key = fname.lower().replace(".csv", "").replace("combined_", "").replace("_", "")
        if key == q:
            return path
    raise ValueError(f"Unknown recording '{name}'. Available: {list(files)}")


def _read(path: Path) -> pd.DataFrame:
    if path.name not in _state:
        df = pd.read_csv(path, on_bad_lines="skip")
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        _state[path.name] = df
    return _state[path.name]


def _features_frame(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in feats:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return out


def _r(x, n=4):
    return None if x is None or pd.isna(x) else round(float(x), n)


# ---------------------------- tools ----------------------------
@mcp.tool()
def list_required_features() -> list[str]:
    """Sensor feature names the fault classifier needs (ch1_/ch2_ vibration features)."""
    return _model()[1]


@mcp.tool()
def list_recordings() -> list[dict]:
    """List the available historical motor recordings and the condition inferred from each filename."""
    return [{"recording": n, "filename_label": _label_from_name(n)} for n in _recordings()]


@mcp.tool()
def classify_fault(readings: dict[str, float]) -> dict:
    """Classify the motor condition from ONE set of sensor readings.
    `readings` must contain all features from list_required_features."""
    pipe, feats = _model()
    missing = [f for f in feats if f not in readings]
    if missing:
        return {"error": f"Missing features: {missing}"}
    X = pd.DataFrame([{f: readings[f] for f in feats}])
    pred = str(pipe.predict(X)[0]).lower().strip()
    out = {"prediction": pred, "guidance": GUIDANCE.get(pred, GUIDANCE["unknown"]), "note": DISCLAIMER}
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        top = sorted(zip(pipe.classes_, proba), key=lambda t: -t[1])[:3]
        out["top_classes"] = {str(c): _r(p) for c, p in top}
        out["model_score"] = _r(proba.max())
    return out


@mcp.tool()
def analyze_recording(recording: str) -> dict:
    """Run the classifier on every sample of a recording (e.g. 'FaultedBearing') and summarise:
    most frequent condition, share of agreeing samples, mean model score, review flag and guidance."""
    pipe, feats = _model()
    path = _resolve(recording)
    X = _features_frame(_read(path), feats)
    preds = pd.Series([str(p).lower().strip() for p in pipe.predict(X)])
    counts = preds.value_counts()
    top = str(counts.index[0])
    share = 100.0 * counts.iloc[0] / len(preds)
    mean_conf = None
    if hasattr(pipe, "predict_proba"):
        mean_conf = float(np.max(pipe.predict_proba(X), axis=1).mean())
    missing = int(X.isna().sum().sum())
    needs_review = share < 70 or (mean_conf is not None and mean_conf < 0.70) or missing > 0
    return {
        "recording": path.name,
        "samples": int(len(preds)),
        "filename_label": _label_from_name(path.name),
        "most_frequent_prediction": top,
        "samples_agreeing_percent": _r(share, 1),
        "prediction_distribution": {str(k): int(v) for k, v in counts.items()},
        "mean_model_score": _r(mean_conf),
        "missing_sensor_cells": missing,
        "human_review_recommended": bool(needs_review),
        "guidance": GUIDANCE.get(top, GUIDANCE["unknown"]),
        "note": DISCLAIMER,
    }


@mcp.tool()
def get_sensor_summary(recording: str) -> dict:
    """Min, median and max of each model sensor feature for a recording."""
    _, feats = _model()
    path = _resolve(recording)
    X = _features_frame(_read(path), feats)
    return {"recording": path.name,
            "features": {f: {"min": _r(X[f].min()), "median": _r(X[f].median()), "max": _r(X[f].max())}
                         for f in feats}}


@mcp.tool()
def compare_to_baseline(recording: str) -> dict:
    """Compare a recording's sensor medians with the healthy Baseline recordings and
    return the 5 features that deviate most (in baseline standard deviations)."""
    _, feats = _model()
    path = _resolve(recording)
    base_frames = [_features_frame(_read(p), feats) for n, p in _recordings().items()
                   if _label_from_name(n) == "baseline" and p != path]
    if not base_frames:
        return {"error": "No baseline recordings available for comparison."}
    base = pd.concat(base_frames, ignore_index=True)
    cur = _features_frame(_read(path), feats)
    rows = []
    for f in feats:
        std = base[f].std()
        dev = (cur[f].median() - base[f].median()) / std if std and std > 0 else np.nan
        rows.append({"feature": f, "recording_median": _r(cur[f].median()),
                     "baseline_median": _r(base[f].median()), "deviation_in_baseline_std": _r(dev, 2)})
    rows = sorted(rows, key=lambda r: -abs(r["deviation_in_baseline_std"] or 0))[:5]
    return {"recording": path.name, "top_deviating_features": rows}


@mcp.tool()
def get_maintenance_guidance(condition: str) -> dict:
    """Maintenance guidance for a condition: baseline, imbalance, bent_shaft, eccentric_rotor,
    faulted_bearing, faulted_coupling."""
    key = condition.lower().strip().replace(" ", "_")
    return {**GUIDANCE.get(key, GUIDANCE["unknown"]), "note": DISCLAIMER}


# ---------------------------- entry ----------------------------
def _self_test():
    print("Features:", list_required_features())
    print("Recordings:", [r["recording"] for r in list_recordings()])
    pipe, feats = _model()
    row = _features_frame(_read(_resolve("FaultedBearing")), feats).dropna().iloc[0].to_dict()
    print("classify_fault ->", classify_fault(row)["prediction"])
    print("analyze_recording ->", analyze_recording("FaultedBearing"))
    print("compare_to_baseline ->", compare_to_baseline("FaultedBearing")["top_deviating_features"][:2])


if __name__ == "__main__":
    if "--test" in sys.argv:
        _self_test()
    else:
        mcp.run()  # stdio; never print to stdout here