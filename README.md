# Tweet Author Identification

An interactive Streamlit app that predicts which author wrote a tweet using
three machine-learning classifiers.

**Try the live app:** [Tweet Author Identification](https://rkstylecode-tweet-author-identification-app-aiaqv1.streamlit.app/)

## Features

- Classify tweet text with Support Vector Machine (SVM), Naive Bayes, or
  Random Forest.
- View the predicted author and each author's prediction confidence.
- Browse and filter the dataset by author or keyword.

## Run locally

### Windows quick start

Double-click `run_app.bat` in the project folder. It creates a virtual
environment if needed, installs requirements, and launches the app.

### PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Keep the
terminal open while using the app; press **Ctrl+C** there to stop it.

## Retrain the models

Replace `tweet_with_authors.csv` with your dataset, then run:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The training script preprocesses the data, trains the models, and overwrites
the `.pkl` artifacts used by the app. The CSV must contain recognizable tweet
text and author/label columns. Restart the app after retraining so it loads
the updated artifacts.

## Project files

- `app.py` — Streamlit application.
- `train_model.py` — data preprocessing, training, and artifact export.
- `tweet_with_authors.csv` — training dataset.
- `vectorizer.pkl`, `model_svm.pkl`, `model_nb.pkl`, `model_rf.pkl` —
  saved feature vectorizer and trained models.
- `clean_df.pkl` — processed dataset displayed in the app.
- `requirements.txt` — pinned Python dependencies.
- `DEPLOYMENT.md` — Streamlit Community Cloud deployment notes.

## Note

The Dataset Viewer displays tweet text and author labels from the included
dataset. Make sure you have permission to share this data before publishing
the repository or app.
