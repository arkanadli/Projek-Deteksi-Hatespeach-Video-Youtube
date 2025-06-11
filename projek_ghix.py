import streamlit as st
import re
import torch
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer
import os
import gdown


st.set_page_config(page_title="Deteksi Hate Speech pada Video", layout="centered")
st.title("🎥 Deteksi Hate Speech dari Video YouTube")

# ====== Ekstrak video ID dari URL ======
def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

# ====== Preprocessing teks ======
def preprocessing(text):
    string = text.lower()
    string = re.sub(r"\n", "", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    string = re.sub(r'[^A-Za-z\s`"]', " ", string)
    return string


@st.cache_resource
def download_model_from_drive():
    file_id = "1OpDWxAl7bcKCm9OVb0vZCmEDZcBi424B"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "model.pth"
    if not os.path.exists(output):
        with st.spinner("📦 Mengunduh model dari Google Drive..."):
            gdown.download(url, output, quiet=False)
    return output

# ====== Label Kelas ======
LABELS = [
    "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
    "HS_Race", "HS_Physical", "HS_Gender", "HS_Other",
    "HS_Weak", "HS_Moderate", "HS_Strong", "PS"
]

# ====== Load model dan tokenizer ======
@st.cache_resource
def load_model_tokenizer():
    model_path = download_model_from_drive()
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobertweet-base-p1")
    return model, tokenizer

# ====== Input URL YouTube ======
youtube_url = st.text_input("Masukkan URL Video YouTube:")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        model, tokenizer = load_model_tokenizer()

        st.info("📥 Mengambil transcript dari video...")
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['id', 'en'])
            full_text = " ".join([entry['text'] for entry in transcript])

            st.success("✅ Transcript berhasil diambil!")
            st.text_area("📄 Cuplikan Transcript:", full_text[:500], height=200)

            # ====== Preprocessing ======
            cleaned_text = preprocessing(full_text)

            # ====== Tokenisasi ======
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)

            # ====== Inference Model ======
            with torch.no_grad():
                outputs = model(**inputs)
                if isinstance(outputs, tuple):  # kalau model return tuple
                    logits = outputs[0]
                else:
                    logits = outputs
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).int().numpy()[0]

            # ====== Tampilkan hasil ======
            st.subheader("📊 Hasil Deteksi Hate Speech:")
            detected = [LABELS[i] for i, val in enumerate(predictions) if val == 1]
            if detected:
                for label in detected:
                    st.write(f"- ✅ **{label}**")
            else:
                st.info("Tidak terdeteksi hate speech.")

        except Exception as e:
            st.error(f"Gagal mengambil transcript: {e}")
    else:
        st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")
