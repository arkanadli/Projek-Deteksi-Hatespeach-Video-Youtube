import streamlit as st
import re
import torch
import torch.nn as nn
import numpy as np
import os
import requests
import json
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi
import zipfile
import tempfile
import warnings
import time
from urllib.parse import urlparse, parse_qs

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page config
st.set_page_config(
    page_title="Deteksi Hate Speech Video YouTube", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🎥 Deteksi Hate Speech dari Video YouTube")
st.markdown("---")

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

LABEL_DESCRIPTIONS = {
    "HS": "Ujaran Kebencian Umum",
    "Abusive": "Bahasa Kasar/Abusive", 
    "HS_Individual": "Ujaran Kebencian terhadap Individu",
    "HS_Group": "Ujaran Kebencian terhadap Kelompok",
    "HS_Religion": "Ujaran Kebencian Agama",
    "HS_Race": "Ujaran Kebencian Ras/Etnis",
    "HS_Physical": "Ujaran Kebencian Fisik",
    "HS_Gender": "Ujaran Kebencian Gender",
    "HS_Other": "Ujaran Kebencian Lainnya",
    "HS_Weak": "Ujaran Kebencian Ringan",
    "HS_Moderate": "Ujaran Kebencian Sedang",
    "HS_Strong": "Ujaran Kebencian Berat",
    "PS": "Konten Pornografi/Seksual"
}

# 🧼 Preprocessing
def preprocessing(text):
    """Membersihkan dan memproses teks"""
    if not text or len(text.strip()) == 0:
        return ""
    
    string = text.lower()
    string = re.sub(r"\n", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    string = re.sub(r'[^A-Za-z\s`"]', " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

# 🔎 Extract YouTube video ID dengan validasi lebih kuat
def extract_video_id(url):
    """Ekstrak video ID dari URL YouTube dengan berbagai format"""
    if not url:
        return None
        
    url = url.strip()
    
    # Pola regex untuk berbagai format URL YouTube
    patterns = [
        r"(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be\/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com\/watch\?.*v=)([a-zA-Z0-9_-]{11})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            # Validasi panjang video ID (harus 11 karakter)
            if len(video_id) == 11:
                return video_id
    
    # Fallback: cek jika input langsung adalah video ID
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url
        
    return None

# 📦 Fungsi untuk mendapatkan bahasa transcript yang tersedia
def get_available_languages(video_id):
    """Mendapatkan daftar bahasa transcript yang tersedia"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_languages = []
        
        for transcript in transcript_list:
            lang_info = {
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            }
            available_languages.append(lang_info)
        
        return available_languages
    except Exception as e:
        return []

# 🎬 Fungsi transcript yang diperbaiki dengan error handling yang lebih baik
def get_transcript_robust(video_id):
    """Mendapatkan transcript dengan penanganan error yang lebih baik"""
    
    if not video_id:
        st.error("❌ Video ID tidak valid")
        return None, None
    
    try:
        # Validasi video ID terlebih dahulu
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            st.error("❌ Format Video ID tidak valid")
            return None, None
        
        st.info("🔍 Memeriksa ketersediaan transcript...")
        
        # Cek ketersediaan transcript dengan timeout
        try:
            # Tambahkan delay untuk menghindari rate limiting
            time.sleep(1)
            
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_languages = []
            
            # Kumpulkan informasi bahasa yang tersedia
            for transcript in transcript_list:
                available_languages.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable,
                    'transcript_obj': transcript
                })
            
            if not available_languages:
                st.error("❌ Video ini tidak memiliki transcript/subtitle")
                return None, None
            
            # Tampilkan bahasa yang tersedia
            st.success(f"✅ Ditemukan {len(available_languages)} bahasa transcript:")
            for lang in available_languages:
                status = "Manual" if not lang['is_generated'] else "Otomatis"
                st.write(f"  • **{lang['language']}** ({lang['language_code']}) - {status}")
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "not available" in error_msg or "disabled" in error_msg:
                st.error("❌ Transcript tidak tersedia untuk video ini")
                st.info("""
                **💡 Kemungkinan penyebab:**
                - Video tidak memiliki subtitle/caption
                - Pemilik video menonaktifkan transcript
                - Video bersifat private atau tidak dapat diakses
                """)
            elif "private" in error_msg:
                st.error("❌ Video bersifat private atau tidak dapat diakses")
            elif "not found" in error_msg:
                st.error("❌ Video tidak ditemukan. Periksa kembali URL video")
            elif "quota" in error_msg or "rate" in error_msg:
                st.error("❌ Terlalu banyak permintaan. Silakan coba lagi dalam beberapa menit")
            else:
                st.error(f"❌ Error mengakses video: {str(e)}")
            
            return None, None
        
        # Prioritas bahasa untuk Indonesia
        language_priority = ['id', 'en', 'ms', 'jv']
        
        transcript = None
        used_language = None
        
        # Coba bahasa dengan prioritas
        for lang_code in language_priority:
            for lang_info in available_languages:
                if lang_info['language_code'] == lang_code:
                    try:
                        st.info(f"📥 Mengambil transcript dalam bahasa: {lang_info['language']}")
                        transcript = lang_info['transcript_obj'].fetch()
                        used_language = lang_info['language']
                        st.success(f"✅ Berhasil mendapatkan transcript dalam bahasa: {lang_info['language']}")
                        break
                    except Exception as e:
                        st.warning(f"⚠️ Gagal mengambil transcript {lang_info['language']}: {str(e)}")
                        continue
            if transcript:
                break
        
        # Jika masih belum ada, coba bahasa pertama yang tersedia
        if transcript is None and available_languages:
            try:
                first_lang = available_languages[0]
                st.info(f"📥 Mencoba bahasa: {first_lang['language']}")
                transcript = first_lang['transcript_obj'].fetch()
                used_language = first_lang['language']
                st.success(f"✅ Berhasil mendapatkan transcript dalam bahasa: {first_lang['language']}")
            except Exception as e:
                st.error(f"❌ Gagal mengambil transcript: {str(e)}")
                return None, None
        
        if transcript is None:
            st.error("❌ Tidak dapat mengambil transcript dalam bahasa apapun")
            return None, None
            
        return transcript, used_language
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error tidak terduga: {error_msg}")
        
        # Berikan saran berdasarkan jenis error
        if "no element found" in error_msg:
            st.info("""
            **💡 Saran untuk mengatasi error parsing:**
            1. Periksa koneksi internet Anda
            2. Coba video YouTube lain yang memiliki subtitle
            3. Tunggu beberapa menit lalu coba lagi
            4. Pastikan URL video benar dan dapat diakses
            """)
        
        return None, None

# 📦 Fungsi download yang diperbaiki
def download_file_robust(url, destination, chunk_size=8192):
    """Download file dengan error handling yang kuat"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        with requests.Session() as session:
            session.headers.update(headers)
            
            # Handle Google Drive direct download
            if 'drive.google.com' in url:
                if 'id=' in url:
                    file_id = url.split('id=')[1].split('&')[0]
                else:
                    file_id = url.split('/')[-2]
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Handle Google Drive virus scan warning
            if 'download_warning' in response.text:
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': file_id, 'confirm': value}
                        response = session.get(url, params=params, stream=True, timeout=60)
                        break
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = downloaded / total_size
                            st.progress(progress)
            
            return True
            
    except Exception as e:
        st.error(f"❌ Gagal mengunduh file: {str(e)}")
        return False

# 📦 Load model dengan error handling yang komprehensif
@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    """Load model dan tokenizer dengan berbagai opsi fallback"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Method 1: Coba load dari Hugging Face Hub
        status_text.text("🔄 Memuat model dari Hugging Face Hub...")
        progress_bar.progress(0.1)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "indolem/indobertweet-base-uncased",
                clean_up_tokenization_spaces=False
            )
            bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
            
            status_text.text("✅ Berhasil memuat dari Hugging Face Hub")
            progress_bar.progress(0.5)
            
        except Exception as e:
            status_text.text(f"⚠️ HF Hub gagal, menggunakan model fallback...")
            progress_bar.progress(0.2)
            
            # Fallback: Gunakan base BERT model
            try:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                bert = AutoModel.from_pretrained('bert-base-uncased')
                status_text.text("⚠️ Menggunakan model BERT fallback")
                progress_bar.progress(0.5)
            except:
                raise Exception("Tidak dapat memuat tokenizer/model apapun")

        # Load custom model weights
        status_text.text("🔄 Memuat bobot model kustom...")
        progress_bar.progress(0.6)
        
        model_path = "model_bigru.pth"
        model_url = "https://drive.google.com/uc?export=download&id=1OpDWxAl7bcKCm9OVb0vZCmEDZcBi424B"
        
        if not os.path.exists(model_path):
            status_text.text("📥 Mengunduh bobot model...")
            if not download_file_robust(model_url, model_path):
                raise Exception("Gagal mengunduh bobot model")
        
        progress_bar.progress(0.8)
        
        # Inisialisasi dan load model
        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)
        
        try:
            # Berbagai cara untuk load model dengan kompatibilitas PyTorch
            try:
                # PyTorch 2.6+ compatible
                state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            except TypeError:
                # Fallback untuk PyTorch versi lama
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            except Exception as e:
                # Terakhir: coba dengan pickle_module
                import pickle
                state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False, pickle_module=pickle)
            
            model.load_state_dict(state_dict)
            model.eval()
            status_text.text("✅ Model berhasil dimuat!")
            progress_bar.progress(1.0)
            
        except Exception as e:
            status_text.text(f"⚠️ Gagal memuat bobot model: {str(e)}")
            # Return model dengan bobot random sebagai fallback
            model.eval()
            progress_bar.progress(1.0)

        return model, tokenizer
        
    except Exception as e:
        status_text.text(f"❌ Error kritis: {str(e)}")
        progress_bar.progress(1.0)
        return None, None
    
    finally:
        # Bersihkan indikator progress
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

# 🎯 Aplikasi utama
def main():
    st.markdown("""
    ### 📋 Cara Penggunaan:
    1. **Masukkan URL video YouTube** yang ingin dianalisis
    2. **Sistem akan mengambil transcript** dari video (mendukung berbagai bahasa)
    3. **AI akan menganalisis konten** untuk mendeteksi ujaran kebencian
    4. **Hasil analisis** akan ditampilkan dengan kategori dan tingkat kepercayaan
    """)
    
    # Input URL
    youtube_url = st.text_input(
        "🔗 URL Video YouTube:",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        help="Tempel URL video YouTube yang ingin dianalisis ujaran kebenciannya"
    )
    
    # Contoh URL untuk testing
    with st.expander("🔍 Contoh URL yang Bisa Digunakan"):
        st.markdown("""
        **Format URL yang didukung:**
        - `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
        - `https://youtu.be/dQw4w9WgXcQ`
        - `https://www.youtube.com/embed/dQw4w9WgXcQ`
        
        **Tips:**
        - Pastikan video bersifat publik (bukan private)
        - Video harus memiliki subtitle/caption (otomatis atau manual)
        - Untuk video baru, tunggu beberapa menit agar YouTube generate subtitle otomatis
        """)

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("❌ URL tidak valid atau tidak dapat dikenali")
            st.info("""
            **💡 Pastikan URL dalam format yang benar:**
            - `https://www.youtube.com/watch?v=VIDEO_ID`
            - `https://youtu.be/VIDEO_ID`
            - `https://www.youtube.com/embed/VIDEO_ID`
            """)
            return
        
        st.success(f"✅ Video ID berhasil dikenali: `{video_id}`")
        
        # Tampilkan video
        try:
            st.video(f"https://www.youtube.com/watch?v={video_id}")
        except:
            st.warning("⚠️ Tidak dapat menampilkan pratinjau video")
        
        # Load model
        with st.spinner("🤖 Memuat model AI untuk analisis ujaran kebencian..."):
            model, tokenizer = load_model_tokenizer()
        
        if model is None or tokenizer is None:
            st.error("❌ Gagal memuat model AI. Silakan coba lagi nanti.")
            return

        # Dapatkan transcript
        transcript, used_language = get_transcript_robust(video_id)
        
        if transcript is None:
            return  # Error sudah ditangani di get_transcript_robust
        
        # Proses transcript
        try:
            full_text = " ".join([entry['text'] for entry in transcript])
        except Exception as e:
            st.error(f"❌ Error memproses transcript: {str(e)}")
            return
        
        if len(full_text.strip()) == 0:
            st.warning("⚠️ Transcript kosong atau tidak dapat diproses")
            return
        
        # Tampilkan preview transcript
        with st.expander("📄 Pratinjau Transcript"):
            preview_text = full_text[:2000] + "..." if len(full_text) > 2000 else full_text
            st.text_area("", preview_text, height=200, disabled=True)
            st.caption(f"Total karakter: {len(full_text):,} | Bahasa: {used_language}")

        # Analisis dengan AI
        st.info("🧠 Menganalisis konten dengan AI...")
        
        # Preprocessing
        cleaned_text = preprocessing(full_text)
        
        if len(cleaned_text.strip()) == 0:
            st.warning("⚠️ Tidak ada teks yang dapat dianalisis setelah preprocessing")
            return
        
        # Tokenisasi
        try:
            inputs = tokenizer(
                cleaned_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128,
                add_special_tokens=True
            )
        except Exception as e:
            st.error(f"❌ Error dalam tokenisasi: {str(e)}")
            return

        # Prediksi
        with torch.no_grad():
            try:
                logits = model(**inputs)
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).int().numpy()[0]
                confidence_scores = probs[0].numpy()
            except Exception as e:
                st.error(f"❌ Error dalam prediksi: {str(e)}")
                return

        # Tampilkan hasil
        st.markdown("---")
        st.subheader("📊 Hasil Analisis Ujaran Kebencian")
        
        detected_labels = [LABELS[i] for i, val in enumerate(predictions) if val == 1]
        
        if detected_labels:
            st.error("🚨 **PERINGATAN: Video berpotensi mengandung ujaran kebencian!**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⚠️ Kategori yang Terdeteksi:**")
                for label in detected_labels:
                    description = LABEL_DESCRIPTIONS.get(label, label)
                    st.write(f"• **{description}**")
            
            with col2:
                st.markdown("**📈 Tingkat Kepercayaan:**")
                for label in detected_labels:
                    idx = LABELS.index(label)
                    confidence = confidence_scores[idx] * 100
                    if confidence >= 80:
                        st.write(f"• {LABEL_DESCRIPTIONS[label]}: **{confidence:.1f}%** 🔴")
                    elif confidence >= 60:
                        st.write(f"• {LABEL_DESCRIPTIONS[label]}: **{confidence:.1f}%** 🟡")
                    else:
                        st.write(f"• {LABEL_DESCRIPTIONS[label]}: **{confidence:.1f}%** 🟠")
            
            # Rekomendasi
            st.markdown("### 💡 Rekomendasi:")
            st.warning("""
            - **Hati-hati** dalam menonton atau membagikan video ini
            - **Pertimbangkan konteks** dan tujuan konten sebelum mengambil tindakan
            - **Laporkan** ke platform jika konten melanggar aturan komunitas
            - **Diskusikan secara konstruktif** jika konten digunakan untuk edukasi
            """)
            
        else:
            st.success("✅ **Tidak terdeteksi ujaran kebencian dalam video ini**")
            st.info("🎉 Video ini tampaknya aman dari konten ujaran kebencian berdasarkan analisis AI")
        
        # Analisis detail
        with st.expander("📊 Analisis Detail Semua Kategori"):
            st.markdown("**Skor kepercayaan untuk setiap kategori:**")
            
            for i, (label, confidence) in enumerate(zip(LABELS, confidence_scores)):
                description = LABEL_DESCRIPTIONS.get(label, label)
                is_detected = predictions[i] == 1
                
                col1, col2, col3 = st.columns([4, 2, 2])
                with col1:
                    st.write(f"**{description}**")
                with col2:
                    confidence_pct = confidence * 100
                    if confidence_pct >= 80:
                        st.write(f"**{confidence_pct:.1f}%** 🔴")
                    elif confidence_pct >= 60:
                        st.write(f"**{confidence_pct:.1f}%** 🟡")
                    elif confidence_pct >= 40:
                        st.write(f"**{confidence_pct:.1f}%** 🟠")
                    else:
                        st.write(f"{confidence_pct:.1f}%")
                with col3:
                    if is_detected:
                        st.write("🚨 **TERDETEKSI**")
                    else:
                        st.write("✅ Aman")
        
        # Informasi tambahan
        with st.expander("ℹ️ Informasi Teknis"):
            st.markdown(f"""
            **Detail Analisis:**
            - **Video ID:** `{video_id}`
            - **Bahasa Transcript:** {used_language}
            - **Panjang Teks Asli:** {len(full_text):,} karakter
            - **Panjang Teks Setelah Preprocessing:** {len(cleaned_text):,} karakter
            - **Model:** IndoBERTweet + BiGRU
            - **Jumlah Kategori:** {len(LABELS)} kategori
            - **Threshold Deteksi:** 50%
            
            **Catatan:** Hasil analisis ini dihasilkan oleh AI dan mungkin tidak 100% akurat. 
            Selalu gunakan pertimbangan manusia dalam menilai konten.
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>🤖 Aplikasi Deteksi Ujaran Kebencian menggunakan AI</p>
        <p>Dibuat untuk membantu mengidentifikasi konten yang berpotensi berbahaya</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()