import streamlit as st
from PIL import Image
from io import BytesIO
import easyocr
import pandas as pd
import sqlite3
import torch
import fitz  # PyMuPDF
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np  # Fix for np not defined

# ---- Page Config ----
st.set_page_config(page_title="Fast Marksheet Correction Web (EasyOCR)", page_icon="📄", layout="wide")

# ---- Dark Theme Styling ----
st.markdown("""
    <style>
    html, body, [class*="st-"] {
        background-color: #1e1e1e;
        color: #f5f5f5;
        font-family: "Segoe UI", sans-serif;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stNumberInput input,
    .stFileUploader,
    .stSlider {
        background-color: #2e2e2e !important;
        color: white !important;
        border-radius: 8px;
        border: 1px solid #555 !important;
    }
    .stButton > button {
        background-color: #444 !important;
        color: white !important;
        border-radius: 6px;
        padding: 8px 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #666 !important;
    }
    .stMetric label, .stMetric div {
        color: white !important;
    }
    header[data-testid="stHeader"] {
        background: none;
    }
    .block-container {
        padding: 1rem 2rem;
    }
    .css-1v0mbdj, .css-1x8cf1d {
        background-color: #2e2e2e !important;
        border-radius: 10px !important;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Load OCR and Embedding Models ----
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return reader, model

reader, bert_model = load_models()

# ---- OCR Function ----
def extract_text_easyocr(image):
    result = reader.readtext(np.array(image), detail=0, paragraph=True)
    return " ".join(result).strip()

# ---- BERT + Fuzzy Similarity ----
def get_similarity(model_answer, student_answer):
    embeddings = bert_model.encode([model_answer, student_answer], convert_to_tensor=True)
    bert_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    fuzzy_score = fuzz.ratio(model_answer, student_answer) / 100.0
    return bert_score, fuzzy_score

# ---- PDF to Image ----
def extract_images_from_pdf(pdf_file):
    images = []
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
    return images

# ---- DB Functions ----
def init_db():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    extracted_answer TEXT,
                    bert_score REAL,
                    fuzzy_score REAL,
                    marks_awarded INTEGER
                )''')
    conn.commit()
    conn.close()

def insert_result(filename, answer, bert_score, fuzzy_score, marks):
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM results WHERE filename = ? AND bert_score = ? AND fuzzy_score = ?", (filename, bert_score, fuzzy_score))
    exists = c.fetchone()[0]
    if not exists:
        c.execute("INSERT INTO results (filename, extracted_answer, bert_score, fuzzy_score, marks_awarded) VALUES (?, ?, ?, ?, ?)",
                  (filename, answer, bert_score, fuzzy_score, marks))
        conn.commit()
    conn.close()

def get_all_results():
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    return df

def delete_selected_rows(row_ids):
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    for row_id in row_ids:
        c.execute("DELETE FROM results WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

# ---- Initialize DB ----
init_db()

# ---- Layout ----
col1, col2 = st.columns([1.2, 3])

with col1:
    st.header("Upload & Settings")
    txt_file = st.file_uploader("Upload Model Answer (.txt)", type=["txt"])
    uploaded_files = st.file_uploader("Upload Answer Sheets (Images or PDFs)", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
    full_mark = st.number_input("Full Marks (for this question)", min_value=1, value=5)
    threshold_high = st.slider("High Similarity Threshold", 0.0, 1.0, 0.75)
    threshold_mid = st.slider("Medium Similarity Threshold", 0.0, 1.0, 0.5)

with col2:
    st.title("⚡ Fast Marksheet Correction Web")

    if txt_file and uploaded_files:
        model_answer = txt_file.read().decode("utf-8")
        results = []

        with st.spinner("Processing..."):
            for file in uploaded_files:
                if file.name.lower().endswith(".pdf"):
                    images = extract_images_from_pdf(file)
                else:
                    try:
                        image = Image.open(BytesIO(file.read())).convert("RGB")
                        images = [image]
                    except Exception as e:
                        st.error(f"Image load failed: {e}")
                        continue

                for idx, image in enumerate(images):
                    display_name = f"{file.name}" if len(images) == 1 else f"{file.name} [Page {idx + 1}]"
                    st.subheader(f"📄 File: {display_name}")

                    try:
                        extracted_text = extract_text_easyocr(image)
                    except Exception as e:
                        st.error(f"OCR failed: {e}")
                        continue

                    st.text_area("✍ Extracted Answer", extracted_text, height=150)

                    bert_score, fuzzy_score = get_similarity(model_answer, extracted_text)
                    st.metric("BERT Score", f"{bert_score:.4f}")
                    st.metric("Fuzzy Score", f"{fuzzy_score:.4f}")

                    if bert_score >= threshold_high and fuzzy_score >= threshold_high:
                        marks = full_mark
                    elif bert_score >= threshold_mid or fuzzy_score >= threshold_mid:
                        marks = int(full_mark / 2)
                    else:
                        marks = 0

                    st.success(f"✅ Marks Awarded: {marks} / {full_mark}")

                    insert_result(display_name, extracted_text, round(bert_score, 4), round(fuzzy_score, 4), marks)

                    results.append({
                        "Filename": display_name,
                        "Extracted Answer": extracted_text,
                        "BERT Score": f"{bert_score:.4f}",
                        "Fuzzy Score": f"{fuzzy_score:.4f}",
                        "Marks Awarded": marks
                    })

        if results:
            df_results = pd.DataFrame(results)
            st.subheader("📊 Current Session Results")
            st.dataframe(df_results)

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Session CSV", csv, "session_results.csv", "text/csv")

            if st.button("🔄 Reset Session Results"):
                results.clear()
                st.experimental_rerun()

    else:
        st.info("👈 Upload model answer and answer sheet images or PDFs to begin.")

    st.subheader("🗃 Stored Results from Database")
    df_all = get_all_results()
    if not df_all.empty:
        selected = st.multiselect("Select row IDs to delete", df_all['id'].tolist())
        if st.button("❌ Delete Selected"):
            delete_selected_rows(selected)
            st.success("Selected entries deleted.")
            df_all = get_all_results()

        st.dataframe(df_all)
        csv_all = df_all.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download All Stored Results", csv_all, "all_results.csv", "text/csv")
    else:
        st.info("No data stored yet.")
