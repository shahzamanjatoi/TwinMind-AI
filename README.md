# TwinMind AI

Motor condition monitoring and fault classification, with AMD GPU evidence. Built for the AMD Developer Hackathon: ACT III (Intelligent Industry track).

**Live app:** [add the Streamlit URL here]  |  **Demo video:** [add the video link here]

TwinMind AI is a Streamlit digital-twin workspace for the Bently Nevada System 1 test-rig recordings (UTK-ASL dataset). It covers six motor conditions: baseline, imbalance, bent shaft, eccentric rotor, faulted bearing and faulted coupling. For any recording it checks input quality, screens for distribution shift, classifies every sample, summarizes a diagnosis with recommended inspection steps, and supports a maintenance workflow (review a flagged sample, record an inspection outcome, save a log).

## What runs where

| Part | Model / method | Where it runs |
|---|---|---|
| Live classifier in the app | scikit-learn pipeline (`models/twinmind_fault_classifier.joblib`, 12 features) | CPU |
| Temporal classifier | PyTorch network (13 features incl. `volts`) | Trained and evaluated on an AMD GPU |
| Sensor screening | Standardized score per row against a full-dataset reference | AMD GPU (saved run) and CPU (live panel) |
| Assistant section | Answers from the saved AMD screening reports; no language model is called | CPU |

## AMD GPU evidence

All GPU work ran in AMD AI Notebooks on an **AMD Radeon Pro W7900D** (PyTorch 2.13.0+rocm10.0.0, ROCm/HIP 7.15.26333). Saved outputs are in `reports/amd_gpu/` and `reports/temporal_gpu/`.

- **Temporal model inference:** 21,137 readings in 7.2 ms on the GPU versus 100.1 ms on the best-tuned CPU setting (13.8x). The default 64-thread CPU run was slower (248.9 ms), so we report the tuned figure. Predicted labels were identical on every row (maximum probability difference 2.4e-7). Timings cover the whole `predict()` call, including preprocessing and data transfer, in a notebook with 128 CPU cores, so treat the ratio as specific to that environment.
- **Sensor screening:** 21,095 rows (24,102 before removing the duplicate `CombinedFaults_02222025.csv` and repeated rows), 12 features, in 0.55 ms on the GPU versus 2.91 ms on the CPU (5.3x). The threshold is the 95th percentile (1.6745), so about 5% of rows are flagged overall by design: 1,055 rows. Compare recordings with each other, for example 914 of 2,898 Bent Shaft rows (31.54%) are flagged.

## Results and limits

- **Temporal model, chronological holdout** (4,286 test records, split by timestamp within each condition): accuracy 96.9%, macro F1 0.971. Weakest class: Eccentric Rotor, recall 81.8% (61 records predicted as baseline and 63 as imbalance). 8 Faulted Coupling records were also predicted as baseline.
- The holdout comes from one recording file, so it shows generalization over time, not performance on a new motor. Validate on independent motors before any operational use.
- The app analyzes historical lab recordings, not a live machine. Predictions and screening flags are decision support and never a confirmed diagnosis.
- Per-recording agreement in the Evaluation tab is expected to be high because each recording holds a single class. It is not a generalization result.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app uses recordings under `data/` if present, otherwise it downloads the public recordings from the UTK-ASL repository. PyTorch is optional: without it the live screening panel runs on the CPU with NumPy. To time the GPU, run the app in an environment with PyTorch + ROCm.

## Repository contents

- `app.py`, `twinmind_panels.py`, `twinmind_theme.py`: the Streamlit app
- `amd_live.py`: live screening panel (PyTorch GPU, or NumPy on the CPU)
- `amd_inference.py`: inference wrapper for the PyTorch temporal model
- `models/`: saved models
- `reports/amd_gpu/`, `reports/temporal_gpu/`: saved GPU runs, evaluation tables and timing evidence
- `maintenance_docs/`, `agent.py`: experimental tool-using assistant scaffold, tested only with a stub model and not used by the app

## Credits

Data: UTK-ASL, Dataset_digital_twin_predictive_maintenance (see that repository for its license). Built with Python, Streamlit, Plotly, scikit-learn, pandas, NumPy and PyTorch (ROCm).
