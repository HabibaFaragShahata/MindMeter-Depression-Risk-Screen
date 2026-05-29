# MindMeter Depression Risk Screen
**Habiba Farag Shehata - 2210018882 **

This is the runnable submission package for MindMeter. It contains the cleaned modelling dataset, an executed end-to-end notebook, ten figures, the four serialised artefacts, and a Flask + JavaScript wizard UI. Everything needed to grade the project end-to-end is in this folder — no external downloads, no missing build steps.

## What MindMeter predicts

The output is a two-layer answer, not a single Yes/No verdict:

1. A **calibrated probability** that the respondent currently reports depression. Calibration uses Platt scaling (`CalibratedClassifierCV(method='sigmoid')`) fit on a held-out slice of the training data so the test split stays untouched.
2. A **risk tier** — one of four bands derived from the calibrated probability with breakpoints at 0.25, 0.50, and 0.75:

   | Band | Range |
   |---|---|
   | Low | 0 – 25 % |
   | Moderate | 25 – 50 % |
   | Elevated | 50 – 75 % |
   | High | 75 – 100 % |

The two-layer design is the central design decision: a screener is most useful when the displayed percentage can be read as a real likelihood instead of an ordinal score.

## Submission layout

```
submission/
├── README.md                       <- this file
├── requirements.txt
├── run.bat                         <- Windows launcher
│
├── notebook.ipynb                  <- executed end-to-end notebook (16 cells, 10 inline figures)
├── app.py                          <- Flask backend
├── templates/index.html            <- stepped-wizard UI with gauge + 4-band strip
│
├── data/depression_clean.csv       <- cleaned modelling dataset (140,700 rows)
│
├── figures/                        <- 10 PNGs
│   ├── 01_target_balance.png
│   ├── 02_age_by_class.png
│   ├── 03_financial_stress.png
│   ├── 04_suicidal_history.png
│   ├── 05_sleep_duration.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_model_comparison.png
│   ├── 08_confusion_matrix.png
│   ├── 09_roc_curve.png
│   └── 10_calibration_curve.png
│
└── deployment artefacts
    ├── mindmeter_model.pkl         <- CalibratedClassifierCV(MLP, method='sigmoid')
    ├── mindmeter_features.pkl      <- ordered list of 17 feature names
    ├── mindmeter_tiers.json        <- 4-band tier map
    └── mindmeter_metadata.json     <- form options, defaults, metrics, comparison table
```

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle Playground Series S4E11 — *Exploring Mental Health Data* |
| Rows | 140,700 |
| Columns | 20 (17 features + `id` + `Name` + target) |
| Target | `Depression` — binary |
| Positive rate | 18.2 % |

Cleaning has already been applied to the shipped `data/depression_clean.csv`.

## Modelling pipeline

```
ColumnTransformer
├── numeric    -> SimpleImputer(median) -> StandardScaler
└── categorical-> SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore', min_frequency=200)
```

Stratified 80/20 split (`random_state=7`). Four-way bake-off (Logistic Regression, Random Forest, Gradient Boosting, MLP). Selection on **ROC-AUC**. Winner wrapped in `CalibratedClassifierCV(method='sigmoid', cv='prefit')` with a 20 % holdout for the calibration fit.

## Results (calibrated MLP, test split)

| Metric | Value |
|---|---:|
| Accuracy | 0.9377 |
| Precision (depression class) | 0.8540 |
| Recall (depression class) | 0.7929 |
| F1 (depression class) | 0.8223 |
| ROC-AUC | 0.9739 |
| Brier score | 0.0471 |

Bake-off at default 0.5 cutoff:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **MLP** | 0.938 | 0.841 | 0.813 | 0.826 | **0.9739** | 0.0452 |
| Logistic Regression | 0.916 | 0.704 | 0.924 | 0.799 | 0.9738 | 0.0618 |
| Gradient Boosting | 0.938 | 0.848 | 0.803 | 0.825 | 0.9737 | 0.0452 |
| Random Forest | 0.912 | 0.700 | 0.908 | 0.790 | 0.9690 | 0.0674 |

Tier distribution on the test set: **80.6 % Low · 2.5 % Moderate · 2.6 % Elevated · 14.2 % High**.

## Web app

`app.py` exposes:

- `GET /` — the wizard UI
- `POST /score` — JSON in, JSON out: `{probability, probability_pct, tier, band_index, tier_labels, tier_edges, breakdown}`

The front-end is a stepped wizard (Profile → Pressure & lifestyle → Background → Result) with a half-circle SVG gauge animated via `requestAnimationFrame` and a 4-row colour-coded tier strip. The result page also lists the 17 input values the model saw, so the prediction is auditable at a glance.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. On Windows, double-clicking `run.bat` does the install and launch in one step.

To re-execute the notebook end-to-end:

```bash
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

## Reproducibility

- One `RANDOM_STATE = 7` constant drives the split, calibration holdout, and every estimator.
- Paths are relative.
- Categorical preprocessing uses `handle_unknown='ignore'`.
- Dependencies are floor-pinned in `requirements.txt`.

## Limitations

- The Kaggle Playground S4E11 dataset is **synthetic** — patterns are designed to mirror a real survey but are not real responses.
- Self-reported. Stress, sleep loss, and substance use are systematically under-reported.
- MindMeter is **not** a clinical instrument and cannot replace a conversation with a qualified mental-health professional.
- Calibration was computed on a held-out slice, not via cross-validation.

---

> **Coursework artefact, not a clinical tool.** If you are struggling, please reach out to a local crisis line — for example, dialing 911 or 988 in the US, or your country's equivalent.
