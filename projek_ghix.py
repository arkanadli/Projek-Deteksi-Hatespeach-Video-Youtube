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
# import safetensors.torch # Mengapa ini tidak digunakan jika Anda tidak memiliki file .safetensors

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
        tokenizer_zip_id = "14xqjAOJDw57wq9XpnYK8sWRQk-KO4uIu"
        tokenizer_dir = "indobertweet-tokenizer"
        tokenizer_zip_url = f"https://drive.google.com/uc?id={tokenizer_zip_id}"

        if not os.path.exists(tokenizer_dir):
            st.info("📥 Downloading tokenizer dari Google Drive...")
            try:
                gdown.download_folder(tokenizer_zip_url, output=tokenizer_dir, quiet=False, use_cookies=False)
            except Exception as e:
                st.warning(f"Gagal mendownload tokenizer lokal: {e}. Menggunakan tokenizer online sebagai fallback.")
                tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
                bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            bert = AutoModel.from_pretrained(tokenizer_dir, local_files_only=True)

        # Jika gagal load dari local, gunakan online
        if 'tokenizer' not in locals() or tokenizer is None:
            st.info("Menggunakan tokenizer online (fallback).")
            tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
            bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")

        # 📦 Download model BiGRU (format .pth)
        model_pth_id = "1OpDWxAl7bcKCm9OVb0vZCmEDZcBi424B"
        pth_path = "model_bigru.pth"
        pth_url = f"https://drive.google.com/uc?id={model_pth_id}"

        if not os.path.exists(pth_path):
            st.info("📥 Downloading model .pth dari Google Drive...")
            try:
                gdown.download(pth_url, pth_path, quiet=False)
                st.success("✅ Model .pth berhasil didownload!")
            except Exception as e:
                st.error(f"❌ Gagal download model .pth: {str(e)}")
                st.info("💡 Pastikan file dapat diakses publik dan ID benar")
                return None, None

        # Initialize model
        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)

        # Load model weights from .pth
        if os.path.exists(pth_path):
            st.info("🔒 Loading model menggunakan PyTorch (.pth)...")
            try:
                # Penting: weights_only=True adalah rekomendasi keamanan untuk PyTorch >= 1.12
                # Jika PyTorch Anda lebih lama, mungkin perlu dihapus atau diatur ke False
                # namun ini dapat menimbulkan risiko keamanan.
                state_dict = torch.load(pth_path, map_location=torch.device('cpu'), weights_only=True)
                model.load_state_dict(state_dict)
                st.success("✅ Model berhasil dimuat dari .pth!")
            except Exception as e:
                st.error(f"❌ Gagal memuat model dari .pth: {str(e)}")
                st.info("💡 Pastikan versi PyTorch Anda kompatibel dan file model tidak korup.")
                return None, None
        else:
            st.error("❌ File model .pth tidak ditemukan. Pastikan model telah diupload ke Google Drive.")
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
        st.markdown(
            """
            1. **Dapatkan API key** dari [SearchAPI.io](https://www.searchapi.io/) (gratis)
            2. **Masukkan API key** di field yang tersedia
            3. **Paste URL video YouTube** yang ingin dianalisis
            4. **Pastikan video memiliki subtitle bahasa Indonesia**
            5. **Klik tombol 'Analisis Video'** dan tunggu prosesnya selesai

            **Catatan**:
            - Meskipun aplikasi ini sekarang mendukung model .pth, **sangat disarankan untuk mengonversi model Anda ke format SafeTensors (.safetensors)** untuk keamanan dan efisiensi.
            - Model akan didownload otomatis dari Google Drive saat pertama kali digunakan.
            - Proses analisis membutuhkan waktu beberapa detik tergantung panjang video.
            """
        )

    # Conversion utility (for manual conversion with PyTorch >=2.6)
    with st.expander("🔧 Panduan Konversi Manual ke SafeTensors"):
        st.markdown(
            """
            **Untuk mengkonversi model .pth ke SafeTensors secara manual:**

            ```python
            # Jalankan di environment dengan PyTorch >=2.6
            from safetensors.torch import save_file
            import torch

            # Load model .pth (aman dengan PyTorch >=2.6)
            # Pastikan path ke model .pth Anda benar
            state_dict = torch.load("model_bigru.pth",
                                            map_location="cpu",
                                            weights_only=True)

            # Save sebagai SafeTensors
            save_file(state_dict, "model_bigru.safetensors")
            print("✅ Konversi berhasil! File model_bigru.safetensors telah dibuat.")
            ```

            **Langkah-langkah:**
            1. Pastikan Anda memiliki PyTorch versi **2.6 atau lebih baru**: `pip install torch>=2.6`
            2. Instal pustaka SafeTensors: `pip install safetensors`
            3. Jalankan script konversi Python di atas di lingkungan lokal Anda.
            4. Setelah `model_bigru.safetensors` dibuat, **upload file tersebut ke Google Drive Anda**.
            5. Dapatkan **ID berbagi** dari file `model_bigru.safetensors` tersebut, lalu **perbarui ID `model_safetensors_id`** dalam kode Streamlit.
            Dengan begitu, aplikasi Anda akan memuat versi SafeTensors yang lebih aman dan direkomendasikan.
            """
        )

        st.warning("⚠️ Menggunakan model .pth mungkin memiliki risiko keamanan jika PyTorch Anda lebih lama dari 1.12 atau jika sumbernya tidak tepercaya. Konversi ke SafeTensors sangat dianjurkan.")

if __name__ == "__main__":
    main()