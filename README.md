# Space Weather Launch-Risk Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E)
![License](https://img.shields.io/badge/License-Educational-blue)

A modular Python application that analyzes historical space-weather events and presents a launch-risk dashboard for educational use.

The project combines pandas-based data processing, rolling 48-hour feature engineering, deterministic risk scoring, a Random Forest classifier, and an interactive Streamlit interface.

**Note:** This tool is an educational demonstration, not an operational launch-authorization or mission-safety system.

---

## Overview

The application processes a unified NASA/DONKI-style space-weather dataset and estimates daily launch risk based on recent activity.

It:

- Cleans and validates space-weather event data
- Calculates solar-flare and geomagnetic-storm features
- Uses a 48-hour rolling observation window
- Produces a score from 0–100
- Categorizes risk as LOW, MODERATE, HIGH, or EXTREME
- Maps risk categories to GO, CAUTION, DELAY, or NO-GO
- Trains a chronological Random Forest model
- Displays interactive charts and daily risk data

---

## Dashboard Features

- Configurable analysis date range
- Overall mission recommendation
- Average risk score and recommendation-day counts
- Daily risk-score trend with threshold markers
- 48-hour X-class, M-class, C-class, and geomagnetic-storm activity
- Filterable daily-results table
- Latest available space-weather conditions
- One-click data rebuild and model retraining

---

## Project Structure

```text
project/
├── app.py                  # Streamlit application
├── data/
│   └── space_weather_unified.csv
├── models/
│   ├── launch_decision_model.pkl
│   └── space_weather_data.pkl
├── src/
│   └── space_weather/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── exploration.py
│       ├── features.py
│       ├── scoring.py
│       ├── model.py
│       ├── artifacts.py
│       ├── visualization.py
│       └── pipeline.py
├── tests/
│   ├── test_pipeline.py
│   └── test_scoring.py
├── pyproject.toml
└── requirements.txt
```

---

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- A local Python environment with pip

---

## Installation

Clone or download the repository, open a terminal in the repository root, and install the project in editable mode:

```powershell
cd project
python -m pip install -e ".[dev]"
```

For a runtime-only installation:

```powershell
python -m pip install -e .
```

---

## Running the Streamlit Application

From the project directory:

```powershell
streamlit run app.py
```

Alternatively, run Streamlit through Python:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

Use the sidebar to select an analysis date range. The **Rebuild data** button cleans the CSV, rebuilds the risk features, retrains the model, refreshes the saved artifacts, and reloads the dashboard.

---

## Running the Command-Line Pipeline

Run the complete non-Streamlit pipeline:

```powershell
python -m space_weather --no-show
```

Customize the dashboard date window:

```powershell
python -m space_weather --start-date 2025-05-01 --end-date 2025-07-09 --no-show
```

Or use the installed command:

```powershell
space-weather --no-show
```

Running the pipeline without `--no-show` will display the matplotlib dashboard.

---

## Modeling Method

### 1. Data cleaning

The raw CSV is:

- Deduplicated by `event_id`
- Completed for selected missing values
- Converted into standardized datetime columns
- Extended with derived flare magnitude and flare class fields
- Used to calculate event duration and temporal components

### 2. Risk feature engineering

For every event date, features are calculated using only prior observations within the preceding 48 hours:

- X-class flare count
- M-class flare count
- C-class flare count
- Maximum Kp index
- Average Kp index
- Storm count
- 24-hour event trend

The current date is excluded from its own feature window.

### 3. Deterministic risk score

| Component | Maximum contribution |
|---|---:|
| X-class flares | 40 |
| M-class flares | 25 |
| Maximum Kp index | 20 |
| Increasing event trend | 15 |

The combined score is capped at 100.

| Risk score | Risk level | Recommendation |
|---:|---|---|
| 0–20 | LOW | GO |
| 21–40 | MODERATE | CAUTION |
| 41–60 | HIGH | DELAY |
| Above 60 | EXTREME | NO-GO |

### 4. Classification model

A `RandomForestClassifier` learns the deterministic `risk_level` label using a chronological split:

- Training records: before 2025-01-01
- Test records: on or after 2025-01-01

The model currently achieves approximately 97% accuracy on the packaged test split. Accuracy describes agreement with the derived risk categories, not real-world launch outcomes.

---

## Dataset

The packaged dataset contains 1,743 space-weather events across July 2023 through July 2025.

It includes solar flares, geomagnetic storms, high-speed streams, timestamps, event classifications, Kp indices, active-region information, observations, instruments, and source locations.

The application uses a snapshot of the source data stored in:

```text
data/space_weather_unified.csv
```

---

## Tests

Run the test suite from the project directory:

```powershell
python -m pytest -p no:cacheprovider tests -q
```

Current tests cover:

- Risk-score capping
- Risk-level boundaries
- Rolling feature generation
- Exclusion of the current date from historical feature windows
- Timezone-safe preprocessing

---

## Output Artifacts

The pipeline generates:

```text
models/launch_decision_model.pkl
models/space_weather_data.pkl
```

- `launch_decision_model.pkl` contains the trained Random Forest classifier.
- `space_weather_data.pkl` contains the latest conditions, feature names, and complete daily risk dataset used by the dashboard.

---

## Known Limitations

- The risk formula is manually designed for education.
- The Random Forest label is derived from the same risk formula.
- The packaged dataset is historical and is not updated automatically.
- A GO or LOW recommendation does not indicate actual mission readiness.
- Additional operational inputs would be required for real launch decisions, including weather, vehicle status, range safety, crew safety, orbital conditions, and official forecasting guidance.

---

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Streamlit
- joblib
- pytest

---

## Contributing

Suggested areas for improvement include:

- Real-time data ingestion
- Automated data refreshes
- Cross-validation and external model evaluation
- Unit tests for all modules
- Downloadable report exports
- Improved accessibility and mobile layouts
- Containerized deployment
- Strictly time-indexed feature validation

---

## Disclaimer

This project is for learning and demonstration purposes only. It must not be used as the sole basis for operational launch decisions or safety-critical actions.


---
## Author

**Anabi Asah**
[LinkedIn](https://linkedin.com/in/anabi-asah) · [GitHub](https://github.com/prof-anabi) · anabiasah@gmail.com