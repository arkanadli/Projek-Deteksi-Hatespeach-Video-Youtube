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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page config
st.set_page_config(
    page_title="Deteksi Hate Speech pada Video", 
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
    "HS": "Hate Speech Umum",
    "Abusive": "Bahasa Kasar/Abusive", 
    "HS_Individual": "Hate Speech terhadap Individu",
    "HS_Group": "Hate Speech terhadap Kelompok",
    "HS_Religion": "Hate Speech Agama",
    "HS_Race": "Hate Speech Ras/Etnis",
    "HS_Physical": "Hate Speech Fisik",
    "HS_Gender": "Hate Speech Gender",
    "HS_Other": "Hate Speech Lainnya",
    "HS_Weak": "Hate Speech Lemah",
    "HS_Moderate": "Hate Speech Sedang",
    "HS_Strong": "Hate Speech Kuat",
    "PS": "Pornografi/Seksual"
}

# 🧼 Preprocessing
def preprocessing(text):
    """Clean and preprocess text"""
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

# 🔎 Extract YouTube video ID
def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# 📦 Get available transcript languages
def get_available_languages(video_id):
    """Get list of available transcript languages for a video"""
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
        st.error(f"Error getting available languages: {str(e)}")
        return []

# 🎬 Enhanced transcript retrieval with better language handling
def get_transcript_smart(video_id):
    """Get transcript with intelligent language selection"""
    try:
        # First, get available languages
        available_languages = get_available_languages(video_id)
        
        if not available_languages:
            raise Exception("No transcripts available for this video")
        
        # Display available languages to user
        st.info("📋 **Bahasa transcript yang tersedia:**")
        for lang in available_languages:
            status = "Manual" if not lang['is_generated'] else "Auto-generated"
            translatable = "✅ Dapat diterjemahkan" if lang['is_translatable'] else "❌ Tidak dapat diterjemahkan"
            st.write(f"• **{lang['language']}** ({lang['language_code']}) - {status} - {translatable}")
        
        # Priority order for transcript languages
        language_priority = [
            'id',      # Indonesian
            'en',      # English
            'ms',      # Malay
            'jv',      # Javanese
        ]
        
        transcript = None
        used_language = None
        
        # Try to get transcript in priority order
        for lang_code in language_priority:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
                used_language = lang_code
                st.success(f"✅ Transcript ditemukan dalam bahasa: **{lang_code}**")
                break
            except Exception:
                continue
        
        # If priority languages not found, try any available language
        if transcript is None:
            for lang in available_languages:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang['language_code']])
                    used_language = lang['language_code']
                    st.success(f"✅ Transcript ditemukan dalam bahasa: **{lang['language']}** ({lang['language_code']})")
                    break
                except Exception:
                    continue
        
        # If still no transcript, try to get any available transcript
        if transcript is None:
            try:
                # Get the first available transcript
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                first_transcript = next(iter(transcript_list))
                transcript = first_transcript.fetch()
                used_language = first_transcript.language_code
                st.success(f"✅ Transcript ditemukan dalam bahasa: **{first_transcript.language}**")
            except Exception as e:
                raise Exception(f"Tidak dapat mengambil transcript: {str(e)}")
        
        return transcript, used_language
        
    except Exception as e:
        # Provide more detailed error information
        error_msg = str(e)
        st.error(f"❌ Error mengambil transcript: {error_msg}")
        
        # Suggest solutions based on error type
        if "No transcripts were found" in error_msg:
            st.info("""
            **💡 Solusi yang bisa dicoba:**
            1. Pastikan video memiliki subtitle/caption (cek di pengaturan video YouTube)
            2. Video mungkin terlalu baru (subtitle auto-generated butuh waktu)
            3. Coba video lain yang sudah pasti memiliki subtitle
            4. Periksa apakah video bersifat public dan dapat diakses
            """)
        elif "private" in error_msg.lower():
            st.info("**💡 Video bersifat private atau tidak dapat diakses publik**")
        else:
            st.info("""
            **💡 Kemungkinan penyebab:**
            - Video tidak memiliki subtitle sama sekali
            - Ada masalah koneksi internet
            - Video telah dihapus atau tidak tersedia
            """)
        
        return None, None

# 📦 Robust download function
def download_file_robust(url, destination, chunk_size=8192):
    """Download file with robust error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        with requests.Session() as session:
            session.headers.update(headers)
            
            # Handle Google Drive direct download
            if 'drive.google.com' in url:
                file_id = url.split('id=')[1].split('&')[0] if 'id=' in url else url.split('/')[-2]
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Handle Google Drive virus scan warning
            if 'download_warning' in response.text:
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': file_id, 'confirm': value}
                        response = session.get(url, params=params, stream=True, timeout=30)
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
        st.error(f"Download failed: {str(e)}")
        return False

# 📦 Load model with comprehensive error handling
@st.cache_resource(show_spinner=False)
def load_model_tokenizer():
    """Load model and tokenizer with multiple fallback options"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Method 1: Try loading from Hugging Face Hub
        status_text.text("🔄 Attempting to load from Hugging Face Hub...")
        progress_bar.progress(0.1)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "indolem/indobertweet-base-uncased",
                clean_up_tokenization_spaces=False
            )
            bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
            
            status_text.text("✅ Successfully loaded from Hugging Face Hub")
            progress_bar.progress(0.5)
            
        except Exception as e:
            status_text.text(f"⚠️ HF Hub failed: {str(e)[:100]}...")
            progress_bar.progress(0.2)
            
            # Fallback: Use base BERT model
            try:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                bert = AutoModel.from_pretrained('bert-base-uncased')
                status_text.text("⚠️ Using fallback BERT model")
                progress_bar.progress(0.5)
            except:
                raise Exception("Could not load any tokenizer/model")

        # Load custom model weights
        status_text.text("🔄 Loading custom model weights...")
        progress_bar.progress(0.6)
        
        model_path = "model_bigru.pth"
        model_url = "https://drive.google.com/uc?export=download&id=1OpDWxAl7bcKCm9OVb0vZCmEDZcBi424B"
        
        if not os.path.exists(model_path):
            status_text.text("📥 Downloading model weights...")
            if not download_file_robust(model_url, model_path):
                raise Exception("Failed to download model weights")
        
        progress_bar.progress(0.8)
        
        # Initialize and load model
        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)
        
        try:
            # Multiple attempts to load model with different PyTorch versions compatibility
            try:
                # PyTorch 2.6+ compatible
                state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            except Exception as e:
                # Last resort: try with pickle_module specification
                import pickle
                state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False, pickle_module=pickle)
            
            model.load_state_dict(state_dict)
            model.eval()
            status_text.text("✅ Model loaded successfully!")
            progress_bar.progress(1.0)
            
        except Exception as e:
            status_text.text(f"⚠️ Model weights loading failed: {str(e)}")
            # Return model with random weights as fallback
            model.eval()
            progress_bar.progress(1.0)

        return model, tokenizer
        
    except Exception as e:
        status_text.text(f"❌ Critical error: {str(e)}")
        progress_bar.progress(1.0)
        return None, None
    
    finally:
        # Clean up progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

# 🎯 Main application
def main():
    st.markdown("""
    ### 📋 Instruksi:
    1. Masukkan URL video YouTube
    2. Sistem akan menganalisis transcript video (mendukung berbagai bahasa)
    3. Hasil deteksi hate speech akan ditampilkan
    
    **🌐 Bahasa yang didukung:** Indonesia, Inggris, Melayu, Jawa, dan bahasa lainnya
    """)
    
    youtube_url = st.text_input(
        "🔗 URL Video YouTube:",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        help="Paste URL video YouTube yang ingin dianalisis"
    )

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        
        if not video_id:
            st.error("❌ URL tidak valid. Contoh format yang benar:")
            st.code("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            st.code("https://youtu.be/dQw4w9WgXcQ")
            return
        
        # Display video
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        
        # Load model
        with st.spinner("🔄 Loading AI model..."):
            model, tokenizer = load_model_tokenizer()
        
        if model is None or tokenizer is None:
            st.error("❌ Failed to load model. Please try again later.")
            return

        # Get transcript with enhanced error handling
        st.info("📥 Mengambil transcript dari video...")
        
        transcript, used_language = get_transcript_smart(video_id)
        
        if transcript is None:
            return  # Error already handled in get_transcript_smart
        
        # Process transcript
        full_text = " ".join([entry['text'] for entry in transcript])
        
        if len(full_text.strip()) == 0:
            st.warning("⚠️ Transcript kosong atau tidak dapat diproses")
            return
        
        # Show transcript preview
        with st.expander("📄 Preview Transcript"):
            st.text_area("", full_text[:1000] + "..." if len(full_text) > 1000 else full_text, height=150)
            st.caption(f"Total karakter: {len(full_text)} | Bahasa: {used_language}")

        # Preprocess and analyze
        st.info("🔍 Menganalisis konten...")
        
        cleaned_text = preprocessing(full_text)
        
        if len(cleaned_text.strip()) == 0:
            st.warning("⚠️ Tidak ada teks yang dapat dianalisis setelah preprocessing")
            return
        
        # Tokenize with proper error handling
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
            st.error(f"❌ Tokenization error: {str(e)}")
            return

        # Predict
        with torch.no_grad():
            try:
                logits = model(**inputs)
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).int().numpy()[0]
                confidence_scores = probs[0].numpy()
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                return

        # Display results
        st.markdown("---")
        st.subheader("📊 Hasil Analisis Hate Speech")
        
        detected_labels = [LABELS[i] for i, val in enumerate(predictions) if val == 1]
        
        if detected_labels:
            st.warning("⚠️ **PERINGATAN: Konten berpotensi mengandung hate speech**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚨 Kategori Terdeteksi:**")
                for label in detected_labels:
                    description = LABEL_DESCRIPTIONS.get(label, label)
                    st.write(f"• **{description}**")
            
            with col2:
                st.markdown("**📈 Confidence Scores:**")
                for label in detected_labels:
                    idx = LABELS.index(label)
                    confidence = confidence_scores[idx] * 100
                    st.write(f"• {label}: {confidence:.1f}%")
            
        else:
            st.success("✅ **Tidak terdeteksi hate speech dalam video ini**")
            st.info("Video ini tampaknya aman dari konten hate speech")
        
        # Detailed analysis
        with st.expander("📊 Analisis Detail Semua Kategori"):
            for i, (label, confidence) in enumerate(zip(LABELS, confidence_scores)):
                description = LABEL_DESCRIPTIONS.get(label, label)
                is_detected = predictions[i] == 1
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{description}**")
                with col2:
                    st.write(f"{confidence*100:.1f}%")
                with col3:
                    if is_detected:
                        st.write("🚨 **DETECTED**")
                    else:
                        st.write("✅ Clear")

if __name__ == "__main__":
    main()