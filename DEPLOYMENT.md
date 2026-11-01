# Deploy to Streamlit Community Cloud

## Prepare the GitHub repository

Create a GitHub repository and upload the app files. Include `app.py`,
`requirements.txt`, and these model artifacts:

- `vectorizer.pkl`
- `model_svm.pkl`
- `model_nb.pkl`
- `model_rf.pkl`
- `clean_df.pkl`

The training CSV and `train_model.py` are only needed if you want to retrain
the models. Do not upload `.venv`; it is excluded by `.gitignore`.

Before pushing, make sure the dataset and model artifacts are okay to share
with the repository's viewers. A private repository can be used if the data
should not be public.

## Create the hosted app

1. Sign in at [share.streamlit.io](https://share.streamlit.io) and connect
   GitHub if prompted.
2. Select **Create app**, then **Yup, I have an app**.
3. Choose the repository and branch, and set the app file path to `app.py`.
4. In **Advanced settings**, choose Python **3.14** to match the environment
   used to create the model artifacts.
5. Select **Deploy**. Streamlit installs the dependencies from
   `requirements.txt`; the app URL will appear when deployment completes.

If dependency versions change, retrain the models with `train_model.py` in the
same Python environment before deploying the new `.pkl` files.
