# TwinMind AI — Self-Explaining Industrial Digital Twin

**Track:** Intelligent Industry
**Project type:** Industrial sensor analysis and maintenance decision support

TwinMind AI is a prototype that helps maintenance teams explore industrial sensor data, identify unusual samples for further review, understand screening results, and record inspection outcomes.

## The problem

Industrial equipment produces sensor readings that can be difficult to interpret quickly. Maintenance teams need practical ways to prioritize unusual readings and decide what to investigate next.

## What TwinMind does

- Loads and explores industrial sensor datasets.
- Provides model-based predictions and sensor-analysis views.
- Screens sensor samples using a statistical anomaly-scoring workflow.
- Displays screening scores and feature statistics.
- Provides suggested inspection steps for flagged samples.
- Records technician review status and notes in a local CSV file.
- Includes saved evidence from a completed AMD ROCm GPU workload.

## Maintenance workflow

1. Select a sensor sample flagged for review.
2. Examine its screening score and source dataset.
3. Inspect the original readings and compare them with expected operating conditions.
4. Follow appropriate site safety procedures.
5. Record the actual inspection outcome and notes.

A statistical alert is not proof of an equipment failure. Technicians must verify suspected issues using the original data and appropriate inspection procedures.

## AMD GPU execution evidence

A sensor-screening workload was executed using PyTorch with ROCm on an **AMD Radeon Pro W7900D** GPU.

Recorded workload results:

| Metric | Result |
|---|---:|
| Sensor samples processed | 45,239 |
| Sensor features processed | 12 |
| Samples flagged for review | 2,262 |
| Screening threshold | 1.7293 |
| Reported GPU workload time | 0.949 seconds |
| Missing or non-finite input values before imputation | 481 |

The timing describes the recorded GPU workload, not the end-to-end Streamlit application. The screening method is a statistical prioritization aid and has not been established as a validated equipment-failure detector.

Evidence and generated outputs are stored under `reports/amd_gpu/`.

## Technology

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Scikit-learn
- PyTorch
- AMD ROCm

## Run locally

1. Install Python and create a virtual environment.
2. Install the project's dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL printed in the terminal.

Some dashboard features require the relevant sensor datasets and model files to be present in their expected project directories.

## Project structure

```text
TwinMind-AI/
├── app.py
├── amd_inference.py
├── models/
├── reports/
│   └── amd_gpu/
├── src/
├── twinmind_sensor_data/
├── evidence/
├── requirements.txt
└── README.md
```

## Limitations

- This is a prototype, not a certified industrial safety system.
- Screening alerts require human verification.
- The saved AMD GPU results are evidence of a previously executed workload; the local dashboard does not automatically run that workload on an AMD GPU.
- Recommendations are general inspection suggestions, not a diagnosis of a particular component.
- Performance results should not be interpreted as measured reductions in downtime or maintenance costs.

## Intended users

Maintenance technicians, reliability engineers, and industrial operations teams who need to investigate unusual equipment sensor readings.

## Hackathon

Built for the AMD Developer Hackathon: ACT III, Track 1 — Intelligent Industry.
