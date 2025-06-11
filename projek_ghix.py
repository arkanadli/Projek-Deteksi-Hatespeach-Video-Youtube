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
from safetensors.torch import load_file # Import load_file untuk SafeTensors

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
        # Pastikan input_ids dan attention_mask ada di device yang sama dengan model
        # Jika model di CPU, ini tidak masalah. Jika nanti di GPU, ini penting.
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

# 📦 Download model dan tokenizer
@st.cache_resource
def load_model_tokenizer():
    try:
        # Tentukan device (CPU atau GPU jika tersedia)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Menggunakan device: {device}")

        # Tokenizer: Menggunakan versi online dari Hugging Face
        st.info("📥 Mengunduh tokenizer dari Hugging Face (online)...")
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        # Model BERT dasar juga perlu dimuat dan dipindahkan ke device
        bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased").to(device)
        st.success("✅ Tokenizer dan model BERT dasar berhasil dimuat!")

        # 📦 Download model BiGRU (format .safetensors)
        model_safetensors_id = "1SfyGkTgRxjx3JEwZ79zJuz5wciOH6d6_" # ID file final_model.safetensors
        safetensors_path = "final_model.safetensors"
        safetensors_url = f"https://drive.google.com/uc?id={model_safetensors_id}"

        if not os.path.exists(safetensors_path):
            st.info("📥 Downloading model SafeTensors dari Google Drive...")
            try:
                gdown.download(safetensors_url, safetensors_path, quiet=False)
                st.success("✅ Model SafeTensors berhasil didownload!")
            except Exception as e:
                st.error(f"❌ Gagal download model SafeTensors: {str(e)}")
                st.info("💡 Pastikan file dapat diakses publik dan ID benar.")
                return None, None, None

        # Initialize model
        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)

        # Load model weights from .safetensors
        if os.path.exists(safetensors_path):
            st.info("🔒 Loading model menggunakan SafeTensors...")
            try:
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict)
                # Pindahkan keseluruhan model ke device yang sama
                model.to(device)
                st.success("✅ Model berhasil dimuat dari SafeTensors dan dipindahkan ke device!")
            except Exception as e:
                st.error(f"❌ Gagal memuat model dari SafeTensors: {str(e)}")
                st.info("💡 Pastikan file SafeTensors tidak korup dan arsitektur model cocok.")
                return None, None, None
        else:
            st.error("❌ File model SafeTensors tidak ditemukan. Pastikan telah diunduh.")
            return None, None, None

        model.eval()
        return model, tokenizer, device # Kembalikan juga device
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {str(e)}")
        # Ini akan memberikan traceback lengkap untuk debugging
        import traceback
        st.exception(traceback.format_exc())
        return None, None, None # Sesuaikan nilai kembalian jika ada error

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
                model, tokenizer, device = load_model_tokenizer() # Tangkap device

            if model is None or tokenizer is None or device is None:
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

                    # Pastikan input tensor berada di device yang sama dengan model
                    # Ambil hanya input_ids dan attention_mask
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    with torch.no_grad():
                        # Lewatkan input_ids dan attention_mask secara eksplisit
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
        st.markdown(
            """
            1. **Dapatkan API key** dari [SearchAPI.io](https://www.searchapi.io/) (gratis)
            2. **Masukkan API key** di field yang tersedia.
            3. **Paste URL video YouTube** yang ingin dianalisis.
            4. **Pastikan video memiliki subtitle bahasa Indonesia**.
            5. **Klik tombol 'Analisis Video'** dan tunggu prosesnya selesai.

            **Catatan**:
            - Tokenizer dan model akan didownload otomatis saat pertama kali digunakan.
            - Proses analisis membutuhkan waktu beberapa detik tergantung panjang video.
            """
        )

    with st.expander("🔧 Panduan Konversi Manual (Jika diperlukan)"):
        st.markdown(
            """
            Jika Anda memiliki model PyTorch dalam format `.pth` dan ingin mengonversinya ke SafeTensors untuk keamanan dan kompatibilitas yang lebih baik:

            ```python
            # Jalankan di environment dengan PyTorch >= 2.6
            from safetensors.torch import save_file
            import torch
            import os

            # Ganti dengan path ke file .pth Anda
            pth_file_path = "nama_model_anda.pth"
            safetensors_output_path = "nama_model_anda.safetensors"

            if not os.path.exists(pth_file_path):
                print(f"Error: File '{pth_file_path}' tidak ditemukan.")
            else:
                try:
                    print(f"Memuat model dari '{pth_file_path}'...")
                    # weights_only=True adalah rekomendasi keamanan untuk PyTorch >= 1.12
                    state_dict = torch.load(pth_file_path, map_location="cpu", weights_only=True)
                    print("Model berhasil dimuat. Mengonversi ke SafeTensors...")
                    save_file(state_dict, safetensors_output_path)
                    print(f"✅ Konversi berhasil! File '{safetensors_output_path}' telah dibuat.")
                    print(f"Sekarang upload '{safetensors_output_path}' ke Google Drive Anda dan perbarui ID-nya di kode Streamlit.")
                except Exception as e:
                    print(f"❌ Gagal mengonversi model: {e}")
                    print("Pastikan file .pth tidak korup dan versi PyTorch Anda terbaru (>= 2.6 direkomendasikan).")
            ```

            **Langkah-langkah:**
            1. Pastikan Anda memiliki PyTorch versi **2.6 atau lebih baru**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
            2. Instal pustaka SafeTensors: `pip install safetensors`
            3. Unduh file `.pth` Anda secara manual ke komputer lokal Anda.
            4. Buat dan jalankan skrip Python di atas (ganti `pth_file_path` dengan nama file `.pth` Anda).
            5. Upload file `.safetensors` yang dihasilkan ke Google Drive dan dapatkan ID berbagi barunya.
            6. Perbarui `model_safetensors_id` di kode Streamlit Anda dengan ID baru tersebut.
            """
        )

if __name__ == "__main__":
    main()