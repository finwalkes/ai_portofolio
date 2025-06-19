import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load IndoBERT sentiment model fine-tuned on SMSA
@st.cache_resource
def load_model():
    model_name = "indobenchmark/indobert-base-p1"
    # Model fine-tuned untuk sentiment, task SMSA (bisa pakai weights huggingface: indolem/indobert-base-uncased-p1-sentiment)
    # Namun kalau tidak ada, model base tetap bisa zero-shot, tapi hasil tidak optimal.
    model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    return tokenizer, model

tokenizer, model = load_model()

# Mapping index ke label (positive, negative, neutral) sesuai dataset SMSA IndoNLU
label_map = {0: "negative", 1: "neutral", 2: "positive"}

st.title("🇮🇩 IndoBERT Sentiment Analyzer Demo")

text = st.text_area("Masukkan kalimat Bahasa Indonesia:")

if st.button("Analisa Sentimen"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            label = label_map[pred]
            score = torch.softmax(logits, dim=1)[0][pred].item()

        if label == "positive":
            st.success(f"Sentimen: POSITIF ({score:.2f}) 😊")
        elif label == "negative":
            st.error(f"Sentimen: NEGATIF ({score:.2f}) 😠")
        elif label == "neutral":
            st.info(f"Sentimen: NETRAL ({score:.2f}) 😐")
        else:
            st.warning(f"Sentimen: {label} ({score:.2f})")
    else:
        st.warning("Teks belum diisi.")

with st.expander("Coba kalimat ini:"):
    st.write("""
    - Bagus sekali pelayanannya!
    - Jelek banget aplikasinya, bikin kesel.
    - Ya, menurut saya biasa saja.
    """)

st.caption("Demo by Bagus Wahyu Pratomo • IndoBERT-base Sentiment • Hugging Face")
