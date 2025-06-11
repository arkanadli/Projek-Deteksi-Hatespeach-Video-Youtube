import streamlit as st
import torch
import torch.nn as nn
import re
import time
import os
import random
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi

# Set page config
st.set_page_config(
    page_title="Deteksi Hate Speech (Indonesia)",
    layout="centered"
)

st.title("🇮🇩 Deteksi Ujaran Kebencian dari Video YouTube (Bahasa Indonesia)")
st.markdown("---")

# Label dan Deskripsi
LABELS = [
    "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
    "HS_Race", "HS_Physical", "HS_Gender", "HS_Other",
    "HS_Weak", "HS_Moderate", "HS_Strong", "PS"
]

LABEL_DESCRIPTIONS = {
    "HS": "Ujaran Kebencian Umum",
    "Abusive": "Bahasa Kasar",
    "HS_Individual": "Kebencian ke Individu",
    "HS_Group": "Kebencian ke Kelompok",
    "HS_Religion": "Kebencian Agama",
    "HS_Race": "Kebencian Ras/Etnis",
    "HS_Physical": "Kebencian Fisik",
    "HS_Gender": "Kebencian Gender",
    "HS_Other": "Kebencian Lainnya",
    "HS_Weak": "Tingkat Ringan",
    "HS_Moderate": "Tingkat Sedang",
    "HS_Strong": "Tingkat Berat",
    "PS": "Konten Seksual"
}

# Model Arsitektur
class IndoBERTweetBiGRU(nn.Module):
    def __init__(self, bert, hidden_size, num_classes):
        super().__init__()
        self.bert = bert
        self.gru = nn.GRU(
            input_size=bert.config.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2 + bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        gru_output, _ = self.gru(sequence_output)
        pooled_output = gru_output.mean(dim=1)
        combined = torch.cat((pooled_output, cls_output), dim=1)
        logits = self.fc(combined)
        return logits

# Preprocessing
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Ekstrak Video ID dari URL
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Ambil transcript otomatis
# Ambil transcript otomatis (dengan dukungan proxy)
def get_transcript(video_id):
    proxies = {
        "https": "http://143.42.66.91:80"
    }
    
    try:
        # Coba ambil transcript pakai proxy
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['id'], proxies=proxies)
        return transcript
    except Exception as e:
        try:
            # Fallback tanpa proxy
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['id'])
            return transcript
        except Exception as fallback_error:
            print("Gagal mengambil transcript:", fallback_error)
            return None


# Load model dan tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
    model = IndoBERTweetBiGRU(bert, hidden_size=512, num_classes=len(LABELS))
    
    # Unduh dan load bobot model
    model_path = "model_bigru.pth"
    if not os.path.exists(model_path):
        st.error("❌ Bobot model tidak ditemukan.")
        return None, None
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, tokenizer

# UI
st.markdown("""
Masukkan URL video YouTube yang memiliki subtitle Bahasa Indonesia.  
Aplikasi ini akan mendeteksi apakah ada ujaran kebencian dalam isi kontennya.
""")

url = st.text_input("🔗 Masukkan URL Video YouTube")

if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("❌ URL tidak valid")
    else:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        st.info("⏳ Mengambil transcript video...")

        with st.spinner("Mengambil transcript Bahasa Indonesia..."):
            transcript = get_transcript(video_id)
        
        if not transcript:
            st.error("❌ Gagal mengambil transcript. Pastikan video memiliki subtitle Bahasa Indonesia dan IP Anda tidak diblokir.")
        else:
            full_text = " ".join([entry["text"] for entry in transcript])
            cleaned_text = preprocessing(full_text)

            if not cleaned_text:
                st.warning("⚠️ Tidak ada teks yang bisa dianalisis.")
            else:
                st.success("✅ Transcript berhasil diambil dan diproses.")
                with st.expander("📄 Pratinjau Teks"):
                    st.text_area("", cleaned_text[:2000], height=200, disabled=True)

                model, tokenizer = load_model_and_tokenizer()
                if model is None:
                    st.stop()

                inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = torch.sigmoid(logits)[0].numpy()
                    preds = (probs > 0.5).astype(int)

                st.subheader("📊 Hasil Analisis")
                detected = [LABELS[i] for i, p in enumerate(preds) if p == 1]

                if detected:
                    st.error("🚨 Ditemukan indikasi ujaran kebencian!")
                    for label in detected:
                        score = probs[LABELS.index(label)] * 100
                        st.write(f"- **{LABEL_DESCRIPTIONS[label]}** — {score:.1f}%")
                else:
                    st.success("✅ Tidak terdeteksi ujaran kebencian dalam video ini.")

                with st.expander("📈 Detail Semua Kategori"):
                    for i, label in enumerate(LABELS):
                        score = probs[i] * 100
                        st.write(f"- {LABEL_DESCRIPTIONS[label]}: {score:.1f}%")

