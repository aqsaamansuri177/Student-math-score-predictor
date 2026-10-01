# 🎓 Student Math Score Predictor

A machine-learning web app that predicts a student's **math exam score** based
on demographic information and other exam results.

Built with **Python · pandas · scikit-learn · Streamlit**.

---

## Project Structure

```
.
├── StudentsPerformance.csv   # Source dataset (1 000 students)
├── train_model.py            # Trains LR + RF, picks the winner, saves artefacts
├── app.py                    # Streamlit web application
├── model.pkl                 # Serialised best model (generated)
├── model_meta.json           # Feature metadata & importances (generated)
└── README.md
```

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.9 |
| pip | 21+ |

---

## Quick Start

### 1 — Clone / place files

Make sure `StudentsPerformance.csv` sits in the same directory as `train_model.py`
and `app.py`.

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install pandas scikit-learn streamlit altair
```

> **Tip:** pin exact versions for reproducibility:
> ```bash
> pip install pandas==2.2.2 scikit-learn==1.5.0 streamlit==1.35.0 altair==5.3.0
> ```

### 4 — Train the models

```bash
python train_model.py
```

Sample output:

```
── Model Comparison ──────────────────────────────
  Linear Regression  │ RMSE: 5.194  │ R²: 0.8763
  Random Forest      │ RMSE: 4.812  │ R²: 0.8932

  ✓  Keeping: Random Forest  (RMSE 4.812, R² 0.8932)
  ✓  Saved model  → model.pkl
  ✓  Saved meta   → model_meta.json
```

> The script automatically selects whichever model achieves the **lower RMSE**
> on the held-out 20 % test set.

### 5 — Launch the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

> **First-run shortcut:** If you skip step 4, the app will train the models
> automatically on first launch.

---

## How It Works

### Models

| Model | Description |
|-------|-------------|
| **Linear Regression** | Fast baseline; uses encoded categorical features + reading/writing scores |
| **Random Forest (200 trees)** | Ensemble of decision trees; captures non-linear interactions |

Both are evaluated on the same 80/20 train/test split (random seed 42).
The model with the lower **Root Mean Squared Error (RMSE)** is saved and used
for predictions.

### Features used

| Feature | Type |
|---------|------|
| Gender | Categorical |
| Race / Ethnicity | Categorical |
| Parental Level of Education | Categorical |
| Lunch Type | Categorical |
| Test Preparation Course | Categorical |
| Reading Score | Numeric (0–100) |
| Writing Score | Numeric (0–100) |

Categorical columns are label-encoded; the encoding map is stored in
`model_meta.json` so the app never needs to re-read the CSV at inference time.

---

## App Features

- **Dropdowns** for all five categorical inputs
- **Sliders** for reading score and writing score
- **Predict button** – instant inference, result clipped to [0, 100]
- **Performance badge** – colour-coded feedback (Excellent / Good / Moderate / Low)
- **Feature Importance chart** – horizontal bar chart (Altair)
- **Model comparison panel** – RMSE & R² for both models
- **Dataset preview** – first 10 rows of the CSV

---

## Retraining

Delete `model.pkl` and `model_meta.json`, then re-run:

```bash
python train_model.py
```

or simply restart the Streamlit app (it auto-trains when artefacts are missing).

---

## License

This project is provided for educational purposes.
Dataset source: [Kaggle – Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).
