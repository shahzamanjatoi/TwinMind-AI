# TwinMind AI — Industrial Motor Intelligence

TwinMind AI turns industrial motor sensor recordings into fault classifications, sensor anomaly screening, and practical maintenance guidance. Built for the **AMD Developer Hackathon: ACT III — Intelligent Industry**, it combines a Streamlit dashboard, machine-learning models, sensor-quality checks, and AMD GPU performance evidence.

**Live app:** [Open TwinMind AI](https://twinmind-ai-mx4nkerrxnf8seuhnvb59t.streamlit.app/)  
**Demo video:** [Watch the project demo](https://drive.google.com/file/d/1XxCxIjy2ujhU9djjTwWRFCgbIUrDA3um/view?usp=sharing)  
**Source code:** [GitHub repository](https://github.com/shahzamanjatoi/TwinMind-AI)

## The problem

Industrial motors generate sensor recordings that contain useful information about their condition. Turning those readings into a clear assessment requires analyzing multiple signals, recognizing fault patterns, and deciding what a technician should inspect next.

TwinMind AI brings these steps together in one workspace. It helps users examine historical motor recordings, identify potentially abnormal readings, understand classification results, and document maintenance reviews.

## What TwinMind AI does

The application works with recordings from the Bently Nevada System 1 test rig in the UTK-ASL dataset.

It supports six motor conditions:

- Baseline
- Imbalance
- Bent shaft
- Eccentric rotor
- Faulted bearing
- Faulted coupling

For an uploaded or selected recording, the application can check input quality, classify sensor samples, screen for unusual sensor patterns, summarize results, and provide inspection guidance. Its maintenance workflow also supports reviewing flagged samples, recording inspection outcomes, and saving a local maintenance log.

## What runs in the system

| Component | Implementation | Execution |
|---|---|---|
| Interactive dashboard | Streamlit, Plotly | CPU |
| Live fault classifier | Scikit-learn pipeline, 12 features | CPU |
| Temporal fault classifier | PyTorch neural network, 13 features | AMD GPU during training and evaluation |
| Sensor anomaly screening | Standardized sensor scores | CPU in the live panel; AMD GPU in saved benchmarks |
| Maintenance assistant | Rule-based guidance using saved reports and maintenance notes | CPU |
| Experimental tool interface | Model, recording, and guidance tools via MCP | Separate experimental server |

The dashboard remains usable without a PyTorch-enabled AMD environment. GPU benchmarking requires a compatible environment with the necessary dependencies.

## AMD GPU execution and evidence

TwinMind AI's GPU experiments were performed in AMD AI Notebooks on an **AMD Radeon Pro W7900D**, using PyTorch 2.13.0 with ROCm 10.0.0 and ROCm/HIP 7.15.26333.

The repository includes saved benchmark reports and evaluation results so that the performance claims can be inspected.

### Measured performance

| Workload | AMD GPU | Tuned CPU | Measured comparison |
|---|---:|---:|---:|
| Temporal model inference, 21,137 readings | 7.2 ms | 100.1 ms | 13.8× faster |
| Sensor screening, 21,095 rows | 0.55 ms | 2.91 ms | 5.3× faster |

The temporal inference benchmark produced identical predicted labels for every row, with a maximum probability difference of \(2.4 \times 10^{-7}\).

These measurements include the relevant prediction pipeline overhead described in the benchmark, including preprocessing and data transfer for temporal inference. They represent a particular notebook environment, not a guaranteed speedup on every machine.

### Model evaluation

The temporal classifier was evaluated on a chronological holdout containing 4,286 records, split by timestamp within each motor condition.

| Metric | Result |
|---|---:|
| Accuracy | 96.9% |
| Macro F1-score | 0.971 |
| Weakest class recall | 81.8% — Eccentric Rotor |

The evaluation also revealed confusion between Eccentric Rotor and Baseline or Imbalance, along with some Faulted Coupling records classified as Baseline. These errors are important considerations when interpreting predictions.

## Sensor screening and maintenance workflow

The sensor-screening panel uses a standardized score calculated against a reference distribution. Its saved benchmark uses the 95th-percentile threshold, which flags approximately 5% of reference rows by design.

A screening flag indicates that a reading deserves further examination; it does not establish that a component has failed.

The maintenance workflow supports the following process:

1. Select a recording and inspect its sensor readings.
2. Review fault classifications and potentially unusual samples.
3. Examine the relevant screening reports and maintenance guidance.
4. Record the result of an actual inspection.
5. Save the maintenance outcome in the project's local log.

This workflow is intended to support technician review rather than replace professional inspection or established site-safety procedures.

## How to run TwinMind AI

### Requirements

- Python 3.10 or a compatible Python version for the project's dependencies
- Git
- The repository's required Python packages
- Internet access on the first run if the public recordings need to be downloaded

For AMD GPU execution, use an environment with compatible AMD GPU hardware, drivers, ROCm, and PyTorch. GPU support is not required for the basic CPU dashboard.

### 1. Clone the repository

```bash
git clone https://github.com/shahzamanjatoi/TwinMind-AI.git
cd TwinMind-AI
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
streamlit run app.py
```

Streamlit will print the local URL. Open it in your browser to use the application.

If the recordings are available under `data/`, the application can use them locally. Otherwise, the project can download the public recordings from the UTK-ASL repository.

For AMD GPU benchmarking, configure PyTorch and ROCm separately according to the target environment.

## Repository contents

| Path | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard |
| `twinmind_panels.py` | Dashboard panel components |
| `twinmind_theme.py` | Dashboard styling |
| `amd_live.py` | Live sensor-screening implementation |
| `amd_inference.py` | PyTorch temporal-model inference wrapper |
| `agent.py` | Maintenance guidance and saved-report assistant |
| `maintenance_docs/` | Maintenance reference material |
| `agent_test.py` | Assistant tests |
| `twinmind_mcp.py` | Experimental MCP tool server |
| `evaluate_amd_model.py` | AMD model evaluation |
| `evaluate_holdout.py` | Chronological holdout evaluation |
| `src/` | Training and diagnostic scripts |
| `models/` | Saved machine-learning models |
| `reports/amd_gpu/` | GPU screening benchmarks and evidence |
| `reports/temporal_gpu/` | Temporal model evaluation and timing evidence |
| `twinmind_sensor_data/` | Sensor recordings used by the application |

## Technical stack

- **Dashboard:** Python, Streamlit, Plotly
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, PyTorch
- **GPU acceleration:** AMD ROCm
- **Development and reproducibility:** Git, Jupyter notebooks
- **Experimental integration:** Model Context Protocol (MCP)

## Important limitations

TwinMind AI is a research and demonstration system, not a certified industrial safety or predictive-maintenance product.

- The application analyzes historical laboratory recordings rather than live-connected industrial machinery.
- The holdout evaluation uses data from one recording file and therefore does not establish generalization to previously unseen motors.
- Sensor screening flags unusual values relative to a reference distribution; the results are not confirmed fault diagnoses.
- Maintenance guidance should be verified by a qualified technician against actual equipment conditions and applicable safety procedures.
- GPU speedups depend on hardware, software versions, workload size, and execution environment.
- The current maintenance assistant uses saved screening reports and rule-based guidance; it does not call a language model in its documented dashboard implementation.

## Future development

Potential next steps include testing on independent motors, integrating live sensor acquisition, improving the weakest fault classes, and validating maintenance recommendations against expert-reviewed outcomes.

## Acknowledgements

- **Dataset:** UTK-ASL, `Dataset_digital_twin_predictive_maintenance`
- **GPU platform:** AMD AI Notebooks
- **Development framework:** AMD ROCm and PyTorch
- **Application libraries:** Streamlit, Plotly, scikit-learn, pandas, and NumPy

Built by **Shahzaman Jatoi** for the AMD Developer Hackathon: ACT III, Intelligent Industry track.
