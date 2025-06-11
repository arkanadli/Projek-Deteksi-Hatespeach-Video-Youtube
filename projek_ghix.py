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
from safetensors.torch import load_file
import yt_dlp
import json
import tempfile
from pathlib import Path
import urllib.parse as urlparse
from urllib.parse import parse_qs


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
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
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

def get_youtube_transcript_ytdlp(video_url_or_id, languages=['id', 'en']):
    try:
        video_id = extract_video_id(video_url_or_id)
        if not video_id:
            video_url = video_url_or_id if video_url_or_id.startswith('http') else f"https://www.youtube.com/watch?v={video_url_or_id}"
            video_id = video_url_or_id
        else:
            video_url = f"https://www.youtube.com/watch?v={video_id}"

        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': languages,
            'subtitlesformat': 'json3',
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

            video_id = info.get('id', video_id)
            title = info.get('title', 'Unknown Title')
            duration = info.get('duration', 0)

            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})

            transcript_data = None
            used_language = None
            is_automatic = False

            for lang in languages:
                if lang in subtitles:
                    transcript_data = subtitles[lang]
                    used_language = lang
                    is_automatic = False
                    break

            if not transcript_data:
                for lang in languages:
                    if lang in automatic_captions:
                        transcript_data = automatic_captions[lang]
                        used_language = lang
                        is_automatic = True
                        break

            if not transcript_data:
                available_subs = list(subtitles.keys()) + list(automatic_captions.keys())
                if available_subs:
                    lang = available_subs[0]
                    if lang in subtitles:
                        transcript_data = subtitles[lang]
                        is_automatic = False
                    else:
                        transcript_data = automatic_captions[lang]
                        is_automatic = True
                    used_language = lang

            if not transcript_data:
                raise Exception("Tidak ada subtitle yang tersedia untuk video ini")

            subtitle_url = None
            for format_info in transcript_data:
                if format_info.get('ext') == 'json3':
                    subtitle_url = format_info.get('url')
                    break

            if not subtitle_url and transcript_data:
                subtitle_url = transcript_data[0].get('url')

            if not subtitle_url:
                raise Exception("URL subtitle tidak ditemukan")

            with urllib.request.urlopen(subtitle_url) as response:
                subtitle_content = response.read().decode('utf-8')

            subtitle_json = json.loads(subtitle_content)
            events = subtitle_json.get('events', [])

            segments = []
            all_texts = []

            for event in events:
                start_time = event.get('tStartMs', 0) / 1000.0
                duration = event.get('dDurationMs', 0) / 1000.0

                text_parts = []
                if 'segs' in event:
                    for seg in event['segs']:
                        if 'utf8' in seg:
                            text_parts.append(seg['utf8'])

                text = ''.join(text_parts).strip()

                if text and text != '\n':
                    segments.append({
                        'start': start_time,
                        'duration': duration,
                        'text': text
                    })
                    all_texts.append(text)

            if not segments:
                raise Exception("Tidak ada teks yang diekstrak dari subtitle")

            full_text = ' '.join(all_texts)

            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            clean_sentences = [re.sub(r'\s+', ' ', sentence).strip()
                               for sentence in sentences if sentence.strip()]

            result = {
                'video_id': video_id,
                'video_url': video_url,
                'title': title,
                'language': used_language,
                'language_code': used_language,
                'transcript_type': 'automatic' if is_automatic else 'manual',
                'is_generated': is_automatic,
                'full_text': full_text,
                'clean_sentences': clean_sentences,
                'segments': segments,
                'total_segments': len(segments),
                'total_sentences': len(clean_sentences),
                'total_duration': duration
            }
            return result

    except Exception as e:
        st.error(f"Gagal mengambil transcript: {str(e)}")
        return None

# 📦 Download model dan tokenizer
@st.cache_resource
def load_model_tokenizer():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Menggunakan device: {device}")

        st.info("📥 Mengunduh tokenizer dari Hugging Face (online)...")
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased").to(device)
        st.success("✅ Tokenizer dan model BERT dasar berhasil dimuat!")

        model_safetensors_id = "1SfyGkTgRxjx3JEwZ79zJuz5wciOH6d6_"
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

        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)

        if os.path.exists(safetensors_path):
            st.info("🔒 Loading model menggunakan SafeTensors...")
            try:
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict)
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
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {str(e)}")
        import traceback
        st.exception(traceback.format_exc())
        return None, None, None

# 🎯 Main App
def main():
    # API Key input - Removed as yt-dlp doesn't require an API key
    # api_key = st.secrets.get("searchapi_key")
    # if not api_key:
    #     api_key = st.text_input(
    #         "🔑 SearchAPI Key",
    #         placeholder="Masukkan API key SearchAPI.io Anda",
    #         type="password",
    #         help="Dapatkan API key gratis di https://www.searchapi.io/"
    #     )

    # YouTube URL input
    youtube_url = st.text_input("🔗 Masukkan URL Video YouTube:")

    if youtube_url: # and api_key: - Removed api_key dependency
        video_id = extract_video_id(youtube_url)
        if video_id:
            # Show video preview
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            # Load model
            with st.spinner("📦 Loading model dan tokenizer..."):
                model, tokenizer, device = load_model_tokenizer()

            if model is None or tokenizer is None or device is None:
                st.error("❌ Gagal memuat model. Periksa koneksi dan file model.")
                return

            if st.button("🚀 Analisis Video"):
                # Get transcript
                with st.spinner("📥 Mengambil transcript dari video..."):
                    transcript_data = get_youtube_transcript_ytdlp(youtube_url)

                if not transcript_data:
                    st.error("❌ Gagal mengambil transcript. Pastikan video memiliki subtitle.")
                    return

                full_text = transcript_data['full_text']
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

                    inputs = {key: val.to(device) for key, val in inputs.items()}

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

    # Instructions
    with st.expander("ℹ️ Cara Menggunakan"):
        st.markdown(
            """
            1. **Paste URL video YouTube** yang ingin dianalisis.
            2. **Pastikan video memiliki subtitle**.
            3. **Klik tombol 'Analisis Video'** dan tunggu prosesnya selesai.

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

# Pastikan Anda telah menginstal library yang diperlukan:
# pip install streamlit transformers torch safetensors yt-dlp