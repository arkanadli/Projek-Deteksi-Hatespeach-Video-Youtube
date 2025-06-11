import streamlit as st
import re
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
import requests
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel

st.set_page_config(
    page_title="Deteksi Hate Speech pada Video", 
    page_icon="🇮🇩",
    layout="centered"
)
st.title("🎥 Deteksi Hate Speech dari Video YouTube")

# ✅ Arsitektur model
class IndoBERTweetBiGRU(nn.Module):
    def __init__(self, bert, hidden_size=512, num_classes=13):
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
        sequence_output = outputs[0]  # or outputs.last_hidden_state
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

LABEL_DESCRIPTIONS = {
    "HS": "Ujaran Kebencian Umum",
    "Abusive": "Bahasa Kasar/Ofensif",
    "HS_Individual": "Kebencian terhadap Individu",
    "HS_Group": "Kebencian terhadap Kelompok",
    "HS_Religion": "Kebencian berbasis Agama",
    "HS_Race": "Kebencian berbasis Ras/Etnis",
    "HS_Physical": "Kebencian berbasis Fisik",
    "HS_Gender": "Kebencian berbasis Gender",
    "HS_Other": "Kebencian Kategori Lain",
    "HS_Weak": "Tingkat Kebencian Ringan",
    "HS_Moderate": "Tingkat Kebencian Sedang",
    "HS_Strong": "Tingkat Kebencian Berat",
    "PS": "Konten Pornografi/Seksual"
}

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
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/watch\?v=([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# 🌐 Ambil transcript dari SearchAPI.io
def get_transcript_from_searchapi(video_id: str, api_key: str) -> Optional[List[Dict]]:
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "youtube_transcripts",
            "video_id": video_id,
            "lang": "id",
            "api_key": api_key
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("transcripts", [])
    except Exception as e:
        st.error(f"Gagal mengambil transcript: {str(e)}")
        return None

# 📦 Download model dan tokenizer dari Google Drive
@st.cache_resource
def load_model_tokenizer():
    try:
        # 📁 Download folder tokenizer (shared as zipped manually)  
        tokenizer_zip_id = "14xqjAOJDw57wq9XpnYK8sWRQk-KO4uIu"  # Ganti dengan ID Google Drive Anda
        tokenizer_dir = "indobertweet-tokenizer"

        if not os.path.exists(tokenizer_dir):
            st.info("📥 Downloading tokenizer dari Google Drive...")
            zip_url = f"https://drive.google.com/uc?id={tokenizer_zip_id}"
            try:
                gdown.download_folder(zip_url, output=tokenizer_dir, quiet=False, use_cookies=False)
            except:
                # Fallback jika download folder gagal
                st.warning("Menggunakan tokenizer online...")
                tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
                bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            bert = AutoModel.from_pretrained(tokenizer_dir, local_files_only=True)

        # Jika gagal load dari local, gunakan online
        if 'tokenizer' not in locals():
            tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
            bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")

        # 📦 Download model BiGRU (.pth)
        model_url_id = "1OpDWxAl7bcKCm9OVb0vZCmEDZcBi424B"  # Ganti dengan ID Google Drive Anda
        model_path = "model_bigru.pth"
        
        if not os.path.exists(model_path):
            st.info("📥 Downloading model dari Google Drive...")
            model_url = f"https://drive.google.com/uc?id={model_url_id}"
            gdown.download(model_url, model_path, quiet=False)

        # Initialize model
        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)
        
        # Load model weights dengan error handling untuk PyTorch security issue
        try:
            # Try safe loading first
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
        except Exception as safe_error:
            try:
                # Fallback to regular loading
                model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            except Exception as regular_error:
                st.error(f"Gagal memuat model: {regular_error}")
                st.info("Pastikan PyTorch versi ≥2.6 atau gunakan safetensors format")
                return None, None
        
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {str(e)}")
        return None, None

# 🎯 Main App
def main():
    # API Key input
    api_key = st.secrets.get("searchapi_key")
    if not api_key:
        api_key = st.text_input(
            "🔑 SearchAPI Key", 
            placeholder="Masukkan API key SearchAPI.io Anda",
            type="password",
            help="Dapatkan API key gratis di https://www.searchapi.io/"
        )

    # YouTube URL input
    youtube_url = st.text_input("🔗 Masukkan URL Video YouTube:")

    if youtube_url and api_key:
        video_id = extract_video_id(youtube_url)
        if video_id:
            # Show video preview
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            
            # Load model
            with st.spinner("📦 Loading model dan tokenizer..."):
                model, tokenizer = load_model_tokenizer()
            
            if model is None or tokenizer is None:
                st.error("❌ Gagal memuat model. Periksa koneksi dan file model.")
                return

            if st.button("🚀 Analisis Video"):
                # Get transcript
                with st.spinner("📥 Mengambil transcript dari video..."):
                    transcript = get_transcript_from_searchapi(video_id, api_key)
                
                if not transcript:
                    st.error("❌ Gagal mengambil transcript. Pastikan video memiliki subtitle bahasa Indonesia.")
                    return

                # Process transcript
                full_text = " ".join([entry['text'] for entry in transcript])
                st.success("✅ Transcript berhasil diambil!")
                
                # Show transcript preview
                with st.expander("📄 Cuplikan Transcript"):
                    st.text_area("", full_text[:1000] + ("..." if len(full_text) > 1000 else ""), height=200)

                # Preprocess and predict
                with st.spinner("🔍 Menganalisis hate speech..."):
                    cleaned_text = preprocessing(full_text)
                    inputs = tokenizer(
                        cleaned_text, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=192
                    )

                    with torch.no_grad():
                        logits = model(**inputs)
                        probs = torch.sigmoid(logits)
                        predictions = (probs > 0.5).int().numpy()[0]

                # Display results
                st.subheader("📊 Hasil Deteksi Hate Speech:")
                detected = [LABELS[i] for i, val in enumerate(predictions) if val == 1]
                
                if detected:
                    st.error("🚨 Terdeteksi hate speech!")
                    for label in detected:
                        prob_score = float(probs[0][LABELS.index(label)] * 100)
                        st.write(f"- 🔴 **{LABEL_DESCRIPTIONS[label]}** ({prob_score:.1f}%)")
                else:
                    st.success("✅ Tidak terdeteksi hate speech")

                # Show detailed scores
                with st.expander("📈 Detail Skor Semua Kategori"):
                    for i, (label, desc) in enumerate(zip(LABELS, [LABEL_DESCRIPTIONS[l] for l in LABELS])):
                        score = float(probs[0][i] * 100)
                        st.write(f"**{desc}**: {score:.1f}%")
        else:
            st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")
    
    elif youtube_url and not api_key:
        st.warning("⚠️ Masukkan API key SearchAPI.io untuk melanjutkan")
    
    # Instructions
    with st.expander("ℹ️ Cara Menggunakan"):
        st.markdown("""
        1. **Dapatkan API key** dari [SearchAPI.io](https://www.searchapi.io/) (gratis)
        2. **Masukkan API key** di field yang tersedia
        3. **Paste URL video YouTube** yang ingin dianalisis
        4. **Pastikan video memiliki subtitle bahasa Indonesia**
        5. **Klik tombol 'Analisis Video'** dan tunggu prosesnya selesai
        
        **Catatan**: 
        - Model akan didownload otomatis dari Google Drive saat pertama kali digunakan
        - Proses analisis membutuhkan waktu beberapa detik tergantung panjang video
        """)

if __name__ == "__main__":
    main()