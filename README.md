# Obesity Level Prediction Dashboard 🧍📊

An interactive Streamlit dashboard that explores an obesity dataset, compares multiple machine
learning models, and lets users enter their own lifestyle/health data to get a real-time obesity
class prediction.

## Live Demo

🔗 [https://obesity-dashboard.streamlit.app/](https://obesity-dashboard.streamlit.app/)

## Features

- **📊 Model Performance** — bar chart comparing accuracy across 5 trained models (GaussianNB,
  SVC, RandomForest, LogisticRegression, KNN).
- **🔍 Confusion Matrix** — interactive heatmap of predictions vs actual classes for any selected
  model.
- **🧪 Predict Your Obesity Class** — fill in a form (age, height, weight, eating habits, activity
  level, transport mode, etc.) and get an instant prediction along with class probabilities and
  calculated BMI.
- **📁 Collected Submissions** — view and download all user-submitted data collected so far.
- **🔄 Retrain Model** — combine the original dataset with newly collected user submissions and
  retrain all models with one click, updating the dashboard's saved model artifacts.

## Project Structure

```
.
├── app.py                                       # Main Streamlit dashboard
├── requirements.txt                              # Python dependencies
├── model_artifacts.joblib                         # Pre-trained models, scaler, results & metadata
├── ObesityDataSet_raw_and_data_sinthetic.csv       # Original training dataset
└── user_submissions.csv                            # Auto-generated: stores user inputs + predictions
```

## Dataset

The dataset contains 2,111 records with 17 attributes covering demographics (Age, Height, Weight,
Gender), eating habits (FAVC, FCVC, NCP, CAEC, CALC), physical condition (SCC, FAF, TUE, CH2O,
SMOKE), and lifestyle (MTRANS, family history of overweight). The target variable
(`NObeyesdad`/`ObesityPrediction`) classifies individuals into 7 categories ranging from
*Insufficient Weight* to *Obesity Type III*.

## Models

| Model | Description |
|---|---|
| GaussianNB | Naive Bayes baseline |
| SVC | Support Vector Classifier (with feature scaling) |
| RandomForestClassifier | Ensemble tree-based model (best performer, ~94% accuracy) |
| LogisticRegression | Linear baseline |
| KNeighborsClassifier | Distance-based classifier |

All models are trained on `StandardScaler`-scaled, one-hot encoded features.

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/JAYANT5VJ/obesity-dashboard.git
cd obesity-dashboard
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## How Data Collection & Retraining Works

1. Every time a user submits the prediction form, their input (plus the predicted class, BMI,
   model used, and timestamp) is appended to `user_submissions.csv`.
2. The **Collected Submissions** tab lets you view/download this data.
3. The **Retrain Model** tab combines the original dataset with `user_submissions.csv`, retrains
   all 5 models, and overwrites `model_artifacts.joblib` with the updated models and results.
4. Refresh the app after retraining to see updated performance metrics.

> **Note:** On free hosting platforms (e.g. Streamlit Community Cloud), the filesystem is
> temporary — `user_submissions.csv` may be lost when the app restarts/sleeps. For permanent
> storage, consider integrating an external database or Google Sheets.

## Deployment

This app can be deployed for free on
[Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repository to GitHub.
2. Go to share.streamlit.io → "New app".
3. Select the repo, branch (`main`), and main file (`app.py`).
4. Click "Deploy".

## Disclaimer

This project is for educational purposes only. Predictions are based on a synthetic/sample dataset
and should **not** be used as medical advice. Always consult a qualified healthcare professional
for health-related concerns.

## License

This project is open-source and available under the MIT License.
