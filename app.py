import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import joblib

# Download NLTK Stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Alignment
# ---------------------------------------------------------
st.set_page_config(page_title="Tweet Author Identification", layout="centered")

st.markdown("""
    <style>
        /* Align selectbox and button vertically */
        .stButton > button {
            margin-top: 28px;
        }
        
        /* Keep the expanded sidebar collapse control permanently visible */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] button {
            visibility: visible !important;
            opacity: 1 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Tweet Author Identification")
st.caption("Machine Learning & NLP Authorship Attribution System")

# ---------------------------------------------------------
# 2. Load Pre-trained Artifacts (Instant Startup)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load('vectorizer.pkl')
    model_svm = joblib.load('model_svm.pkl')
    model_nb = joblib.load('model_nb.pkl')
    model_rf = joblib.load('model_rf.pkl')
    df = joblib.load('clean_df.pkl')
    return vectorizer, model_svm, model_nb, model_rf, df

vectorizer, model_svm, model_nb, model_rf, df = load_artifacts()

# Text Preprocessing (same function used during training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------------------------------------
# 3. Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Go to:", ["Live Classification", "Dataset Viewer"])

# ---------------------------------------------------------
# TAB 1: Live Classification
# ---------------------------------------------------------
if app_mode == "Live Classification":
    st.markdown("### Live Classification")
    user_input = st.text_area("Enter a tweet text to identify its author:", value="Show me your 🌸 FLOWER POWER 🌸 with my new #SmallTalk filter on instagram...")

    col1, col2 = st.columns([2, 1])
    with col1:
        model_choice = st.selectbox("Algorithm", ["SVM", "Naïve Bayes", "Random Forest"])
    with col2:
        predict_btn = st.button("Predict Author", type="primary", use_container_width=True)

    if predict_btn:
        if user_input.strip() != "":
            processed = clean_text(user_input)
            vector = vectorizer.transform([processed])
            
            if model_choice == "SVM":
                pred = model_svm.predict(vector)[0]
                probs = model_svm.predict_proba(vector)[0]
                classes = model_svm.classes_
            elif model_choice == "Naïve Bayes":
                pred = model_nb.predict(vector)[0]
                probs = model_nb.predict_proba(vector)[0]
                classes = model_nb.classes_
            else:
                pred = model_rf.predict(vector)[0]
                probs = model_rf.predict_proba(vector)[0]
                classes = model_rf.classes_
                
            st.success(f"**Predicted Author:** {pred}")
            
            res_df = pd.DataFrame({"Author": classes, "Confidence": probs}).sort_values(by="Confidence", ascending=False).reset_index(drop=True)
            st.dataframe(res_df.style.format({"Confidence": "{:.2%}"}), use_container_width=True)
        else:
            st.warning("Please enter text first.")

# ---------------------------------------------------------
# TAB 2: Dataset Viewer
# ---------------------------------------------------------
elif app_mode == "Dataset Viewer":
    st.markdown("### Dataset Section")
    st.caption(f"Inspecting the loaded raw dataset (`tweet_with_authors.csv`). Total Records: **{len(df)}**")
    
    search_query = st.text_input("Filter by Author or Keyword:")
    
    display_df = df[["author", "tweet"]]
    if search_query:
        display_df = display_df[
            display_df['author'].str.contains(search_query, case=False, na=False) | 
            display_df['tweet'].str.contains(search_query, case=False, na=False)
        ]
        
    st.dataframe(display_df.head(100), use_container_width=True)
    