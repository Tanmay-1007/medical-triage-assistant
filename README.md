# Tiny Medical Triage Assistant

Hybrid **rule-based + ML** triage demo built with **Python + Streamlit**.

⚠️ **Educational tool only. Not for real medical use or diagnosis.**

## Features

- Rule-based triage:
  - Uses symptoms, red-flag signs, age, and duration.
  - Outputs: Emergency / Urgent / Routine / Self-care.
- Optional ML model (Random Forest):
  - Trained on synthetic data + logged cases.
  - Provides probability for each class.
- Streamlit UI:
  - Sidebar inputs, center result, right-side EDA.
  - Dark theme, minimal design.
- Logging:
  - Saves anonymized cases to `logs/triage_logs.csv`.
  - Downloadable from UI.

## Local Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
python data_gen.py
python train_model.py    # optional but recommended
streamlit run app.py
