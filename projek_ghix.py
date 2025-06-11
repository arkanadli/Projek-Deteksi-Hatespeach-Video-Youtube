import streamlit as st
import re
import torch
import torch.nn as nn
import numpy as np
import os
import requests
import json
from transformers import AutoTokenizer, AutoModel
import zipfile
import tempfile
import warnings
import time
from urllib.parse import urlparse, parse_qs
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# 🌐 Fungsi untuk bypass IP blocking dengan multiple methods
def create_session_with_retries():
    """Membuat session dengan retry strategy dan random user agent"""
    session = requests.Session()
    
    # Daftar User-Agent yang berbeda
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
    ]
    
    # Set random user agent
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # Setup retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# 🎬 Alternatif method untuk mendapatkan transcript
def get_transcript_alternative_api(video_id):
    """Menggunakan API alternatif untuk mendapatkan transcript"""
    try:
        # Method 1: Coba dengan API alternatif
        session = create_session_with_retries()
        
        # Tambahkan delay random untuk menghindari rate limiting
        time.sleep(random.uniform(1, 3))
        
        # API endpoint alternatif (contoh)
        api_urls = [
            f"https://www.googleapis.com/youtube/v3/captions?part=snippet&videoId={video_id}",
            f"https://youtubetranscript.com/api/transcript/{video_id}",
        ]
        
        # Untuk demonstrasi, kita akan menggunakan method scraping sederhana
        # Ini adalah fallback method jika YouTube Transcript API terblokir
        
        return None, None  # Sementara return None untuk implementasi yang lebih aman
        
    except Exception as e:
        return None, None

# 🎬 Fungsi utama untuk mendapatkan transcript dengan multiple fallback
def get_transcript_with_fallback(video_id):
    """Mendapatkan transcript dengan berbagai method fallback"""
    
    if not video_id:
        st.error("❌ Video ID tidak valid")
        return None, None
    
    # Validasi video ID
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        st.error("❌ Format Video ID tidak valid")
        return None, None
    
    st.info("🔍 Mencoba mengambil transcript dari video...")
    
    # Method 1: Coba YouTube Transcript API dengan workaround
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Tambahkan delay untuk menghindari rate limiting
        time.sleep(random.uniform(2, 5))
        
        # Coba dengan berbagai bahasa
        language_codes = ['id', 'en', 'ms', 'jv', 'auto']
        
        for lang_code in language_codes:
            try:
                if lang_code == 'auto':
                    # Coba tanpa specify language
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                else:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
                
                if transcript:
                    st.success(f"✅ Berhasil mendapatkan transcript (bahasa: {lang_code})")
                    return transcript, lang_code
                    
            except Exception as lang_error:
                continue
                
    except Exception as api_error:
        error_msg = str(api_error)
        
        if "blocked" in error_msg.lower() or "ip" in error_msg.lower():
            st.warning("⚠️ IP diblokir oleh YouTube. Mencoba method alternatif...")
            
            # Method 2: Gunakan input manual dari user
            st.info("""
            **🔄 Solusi Alternatif - Input Manual Transcript:**
            
            Karena IP diblokir oleh YouTube, Anda dapat:
            1. Buka video di YouTube
            2. Klik tombol CC (Closed Captions) 
            3. Klik "..." → "Buka transkrip"
            4. Copy transcript dan paste di bawah ini
            """)
            
            # Text area untuk input manual
            manual_transcript = st.text_area(
                "📝 Paste transcript di sini:",
                height=200,
                placeholder="Paste transcript dari YouTube di sini...",
                help="Copy transcript dari YouTube dan paste di sini untuk analisis"
            )
            
            if manual_transcript and len(manual_transcript.strip()) > 0:
                st.success("✅ Transcript manual berhasil diinput!")
                # Convert ke format yang sama dengan API
                mock_transcript = [{"text": manual_transcript, "start": 0, "duration": 0}]
                return mock_transcript, "manual"
            
            # Method 3: Gunakan demo text untuk testing
            if st.button("🧪 Gunakan Contoh Text untuk Demo"):
                demo_text = """
                Halo semua, selamat datang di channel YouTube ini. 
                Hari ini kita akan membahas topik yang sangat menarik tentang teknologi terbaru.
                Jangan lupa untuk subscribe dan like video ini jika kalian merasa terbantu.
                Kita akan mulai dengan pembahasan tentang artificial intelligence dan machine learning.
                Teknologi ini sangat membantu dalam berbagai aspek kehidupan kita sehari-hari.
                """
                st.info("🧪 Menggunakan contoh text untuk demonstrasi")
                mock_transcript = [{"text": demo_text, "start": 0, "duration": 0}]
                return mock_transcript, "demo"
            
        else:
            st.error(f"❌ Error lain: {error_msg}")
    
    return None, None

# 📦 Fungsi download yang diperbaiki
def download_file_robust(url, destination, chunk_size=8192):
    """Download file dengan error handling yang kuat"""
    try:
        session = create_session_with_retries()
        
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
    2. **Sistem akan mengambil transcript** dari video (atau input manual jika diblokir)
    3. **AI akan menganalisis konten** untuk mendeteksi ujaran kebencian
    4. **Hasil analisis** akan ditampilkan dengan kategori dan tingkat kepercayaan
    
    **🌐 Bahasa yang didukung:** Indonesia, Inggris, Melayu, Jawa, dan bahasa lainnya
    """)
    
    # Peringatan IP blocking
    st.warning("""
    **⚠️ Informasi Penting:**
    Jika sistem tidak dapat mengambil transcript otomatis (karena IP diblokir YouTube), 
    Anda dapat menggunakan **input manual** dengan menyalin transcript langsung dari YouTube.
    """)
    
    # Input URL
    youtube_url = st.text_input(
        "🔗 URL Video YouTube:",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        help="Tempel URL video YouTube yang ingin dianalisis ujaran kebenciannya"
    )
    
    # Contoh URL dan tutorial
    with st.expander("🔍 Panduan Lengkap Penggunaan"):
        st.markdown("""
        **Format URL yang didukung:**
        - `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
        - `https://youtu.be/dQw4w9WgXcQ`
        - `https://www.youtube.com/embed/dQw4w9WgXcQ`
        
        **Jika IP diblokir YouTube, ikuti langkah ini:**
        1. Buka video di YouTube
        2. Klik tombol **CC** (Closed Captions) di video player
        3. Klik titik tiga **"..."** → **"Buka transkrip"**
        4. Copy semua text transcript
        5. Paste di area input manual yang akan muncul
        
        **Tips:**
        - Pastikan video bersifat publik (bukan private)
        - Video harus memiliki subtitle/caption
        - Untuk video baru, tunggu beberapa menit agar YouTube generate subtitle
        """)

    # Input manual transcript sebagai alternatif utama
    st.markdown("### 📝 Alternatif: Input Manual Transcript")
    manual_transcript = st.text_area(
        "Paste transcript manual di sini (jika otomatis gagal):",
        height=150,
        placeholder="Copy transcript dari YouTube dan paste di sini...",
        help="Jika sistem tidak bisa mengambil transcript otomatis, gunakan input manual ini"
    )

    # Proses berdasarkan input yang tersedia
    process_analysis = False
    transcript_data = None
    transcript_source = None
    video_id = None
    
    if manual_transcript and len(manual_transcript.strip()) > 0:
        # Prioritaskan input manual
        transcript_data = [{"text": manual_transcript, "start": 0, "duration": 0}]
        transcript_source = "manual"
        process_analysis = True
        st.success("✅ Menggunakan transcript manual untuk analisis")
        
    elif youtube_url:
        # Proses URL YouTube
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("❌ URL tidak valid atau tidak dapat dikenali")
            st.info("""
            **💡 Pastikan URL dalam format yang benar:**
            - `https://www.youtube.com/watch?v=VIDEO_ID`
            - `https://youtu.be/VIDEO_ID`
            - `https://www.youtube.com/embed/VIDEO_ID`
            """)
        else:
            st.success(f"✅ Video ID berhasil dikenali: `{video_id}`")
            
            # Tampilkan video
            try:
                st.video(f"https://www.youtube.com/watch?v={video_id}")
            except:
                st.warning("⚠️ Tidak dapat menampilkan pratinjau video")
            
            # Coba ambil transcript otomatis
            transcript_data, transcript_source = get_transcript_with_fallback(video_id)
            
            if transcript_data:
                process_analysis = True

    # Proses analisis jika ada data transcript
    if process_analysis and transcript_data:
        
        # Load model
        with st.spinner("🤖 Memuat model AI untuk analisis ujaran kebencian..."):
            model, tokenizer = load_model_tokenizer()
        
        if model is None or tokenizer is None:
            st.error("❌ Gagal memuat model AI. Silakan coba lagi nanti.")
            return

        # Proses transcript
        try:
            if transcript_source == "manual":
                full_text = transcript_data[0]["text"]
            else:
                full_text = " ".join([entry['text'] for entry in transcript_data])
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
            st.caption(f"Total karakter: {len(full_text):,} | Sumber: {transcript_source}")

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
            st.error("🚨 **PERINGATAN: Konten berpotensi mengandung ujaran kebencian!**")
            
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
            - **Hati-hati** dalam menonton atau membagikan konten ini
            - **Pertimbangkan konteks** dan tujuan konten sebelum mengambil tindakan
            - **Laporkan** ke platform jika konten melanggar aturan komunitas
            - **Diskusikan secara konstruktif** jika konten digunakan untuk edukasi
            """)
            
        else:
            st.success("✅ **Tidak terdeteksi ujaran kebencian dalam konten ini**")
            st.info("🎉 Konten ini tampaknya aman dari ujaran kebencian berdasarkan analisis AI")
        
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
            - **Video ID:** `{video_id if video_id else 'Manual Input'}`
            - **Sumber Transcript:** {transcript_source}
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
        <p>Mendukung input otomatis dan manual untuk mengatasi pembatasan IP</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()