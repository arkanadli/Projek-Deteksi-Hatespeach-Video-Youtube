import streamlit as st
import re
import torch
import torch.nn as nn
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModel
import os

st.set_page_config(page_title="Deteksi Hate Speech pada Video", layout="centered")
st.title("🎥 Deteksi Hate Speech dari Video YouTube")

# ✅ Arsitektur model
class IndoBERTweetBiGRU(nn.Module):
    def __init__(self, bert, hidden_size, num_classes):
        super().__init__()
        self.bert = bert
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2 + self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]  # [CLS] token
        gru_output, _ = self.gru(sequence_output)
        pooled_output = gru_output.mean(dim=1)
        combined = torch.cat((pooled_output, cls_output), dim=1)
        logits = self.fc(combined)
        return logits

# 🔠 Label klasifikasi
LABELS = [
    "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
    "HS_Race", "HS_Physical", "HS_Gender", "HS_Other",
    "HS_Weak", "HS_Moderate", "HS_Strong", "PS"
]

# 🧼 Preprocessing
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

# 🔎 Ambil ID dari URL YouTube
def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

# 📦 Load model dan tokenizer
@st.cache_resource
def load_model_tokenizer():
    MODEL_DIR = os.path.join(os.getcwd(), "indobertweet-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    bert = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)

    model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, tokenizer

# 🎯 Input dari user
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

            # 🧹 Preprocessing
            cleaned_text = preprocessing(full_text)

            # 🧪 Tokenisasi (maksimal 192 token)
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=192)

            # 🔍 Inference
            with torch.no_grad():
                logits = model(**inputs)
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).int().numpy()[0]

            # 📊 Tampilkan hasil
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
