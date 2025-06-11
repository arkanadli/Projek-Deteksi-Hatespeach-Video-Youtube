import streamlit as st
import torch
import torch.nn as nn
import re
import time
import os
import logging
from typing import Optional, List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Deteksi Hate Speech Indonesia",
    page_icon="🇮🇩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Constants
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

LABEL_COLORS = {
    "HS": "🔴", "Abusive": "🟠", "HS_Individual": "🔴", "HS_Group": "🔴",
    "HS_Religion": "🔴", "HS_Race": "🔴", "HS_Physical": "🔴", "HS_Gender": "🔴",
    "HS_Other": "🔴", "HS_Weak": "🟡", "HS_Moderate": "🟠", "HS_Strong": "🔴", "PS": "🟣"
}

# Proxy configuration
PROXY_CONFIG = {
    "host": "207.244.217.165",
    "port": "6712",
    "user": "ktsafopf",
    "pass": "rlqzqqwoytk8"
}

class IndoBERTweetBiGRU(nn.Module):
    """
    Enhanced IndoBERTweet model with Bidirectional GRU for hate speech detection
    """
    def __init__(self, bert, hidden_size: int = 512, num_classes: int = 13, dropout_rate: float = 0.3):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(
            input_size=bert.config.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if hidden_size > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2 + bert.config.hidden_size, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_size * 2 + bert.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        
        # BiGRU processing
        sequence_output = self.dropout(sequence_output)
        gru_output, _ = self.gru(sequence_output)
        pooled_output = gru_output.mean(dim=1)
        
        # Combine features
        combined = torch.cat((pooled_output, cls_output), dim=1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        logits = self.fc(combined)
        return logits

def preprocessing(text: str) -> str:
    """
    Enhanced text preprocessing for Indonesian text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers and special characters, keep Indonesian letters
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats
    """
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

def get_proxy_config() -> Dict[str, str]:
    """
    Get proxy configuration
    """
    proxy_url = f"http://{PROXY_CONFIG['user']}:{PROXY_CONFIG['pass']}@{PROXY_CONFIG['host']}:{PROXY_CONFIG['port']}"
    return {
        "http": proxy_url,
        "https": proxy_url
    }

def get_transcript(video_id: str, max_retries: int = 3) -> Optional[List[Dict]]:
    """
    Get YouTube transcript with improved error handling and retry mechanism
    """
    proxies = get_proxy_config()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to get transcript (attempt {attempt + 1}/{max_retries})")
            
            # Check if transcript is available
            response = requests.get(
                f"https://video.google.com/timedtext?lang=id&v={video_id}",
                proxies=proxies,
                timeout=15,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Transcript check failed with status {response.status_code}")
                continue
                
            if "<transcript>" not in response.text:
                logger.warning("No transcript found in response")
                continue
            
            # Get the actual transcript
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['id', 'id-ID'], 
                proxies=proxies
            )
            
            logger.info("Transcript retrieved successfully")
            return transcript
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    logger.error("All attempts to get transcript failed")
    return None

@st.cache_resource
def load_model_and_tokenizer() -> Tuple[Optional[IndoBERTweetBiGRU], Optional[AutoTokenizer]]:
    """
    Load pre-trained model and tokenizer with caching
    """
    try:
        logger.info("Loading tokenizer and model...")
        
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
        model = IndoBERTweetBiGRU(bert, hidden_size=512, num_classes=len(LABELS))
        
        model_path = "model_bigru.pth"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None
            
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

def predict_hate_speech(text: str, model: IndoBERTweetBiGRU, tokenizer: AutoTokenizer) -> Tuple[List[str], List[float]]:
    """
    Predict hate speech categories for given text
    """
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256  # Increased max length
        )
        
        with torch.no_grad():
            logits = model(**inputs)
            probs = torch.sigmoid(logits)[0].numpy()
            preds = (probs > 0.5).astype(int)
        
        detected = [LABELS[i] for i, p in enumerate(preds) if p == 1]
        scores = [float(prob * 100) for prob in probs]
        
        return detected, scores
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return [], []

def display_results(detected_labels: List[str], all_scores: List[float]):
    """
    Display prediction results with improved formatting
    """
    if detected_labels:
        st.error("🚨 **PERINGATAN: Terdeteksi indikasi ujaran kebencian!**")
        
        st.markdown("### 📋 Kategori yang Terdeteksi:")
        for label in detected_labels:
            score = all_scores[LABELS.index(label)]
            emoji = LABEL_COLORS.get(label, "⚪")
            st.markdown(f"{emoji} **{LABEL_DESCRIPTIONS[label]}** — `{score:.1f}%`")
    else:
        st.success("✅ **Tidak terdeteksi ujaran kebencian dalam video ini.**")
        st.balloons()

def main():
    """
    Main application function
    """
    # Header
    st.title("🇮🇩 Deteksi Ujaran Kebencian dari Video YouTube")
    st.markdown("### *Analisis Konten Bahasa Indonesia*")
    st.markdown("---")
    
    # Description
    st.markdown("""
    **Fitur Aplikasi:**
    - 🎥 Menganalisis video YouTube dengan subtitle Bahasa Indonesia
    - 🤖 Menggunakan model IndoBERTweet + BiGRU untuk deteksi akurat
    - 📊 Mendeteksi 13 kategori ujaran kebencian yang berbeda
    - 🔍 Memberikan skor kepercayaan untuk setiap kategori
    
    **Cara Penggunaan:**
    Masukkan URL video YouTube yang memiliki subtitle Bahasa Indonesia di kolom di bawah ini.
    """)
    
    # Input URL
    url = st.text_input(
        "🔗 **Masukkan URL Video YouTube:**",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Pastikan video memiliki subtitle Bahasa Indonesia"
    )
    
    if url:
        video_id = extract_video_id(url)
        
        if not video_id:
            st.error("❌ **URL tidak valid.** Pastikan Anda memasukkan URL YouTube yang benar.")
            st.info("Contoh URL yang valid: https://www.youtube.com/watch?v=VIDEO_ID")
            return
        
        # Display video
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        
        # Process button
        if st.button("🚀 **Analisis Video**", type="primary"):
            with st.spinner("⏳ Mengambil dan memproses transcript..."):
                transcript = get_transcript(video_id)
            
            if not transcript:
                st.error("""
                ❌ **Gagal mengambil transcript.**
                
                **Kemungkinan penyebab:**
                - Video tidak memiliki subtitle Bahasa Indonesia
                - Video bersifat privat atau terbatas
                - Masalah koneksi jaringan
                - IP address diblokir oleh YouTube
                
                **Solusi:**
                - Pastikan video memiliki subtitle Indonesia (CC)
                - Coba video lain yang memiliki subtitle
                - Tunggu beberapa saat dan coba lagi
                """)
                return
            
            # Process transcript
            full_text = " ".join([entry["text"] for entry in transcript])
            cleaned_text = preprocessing(full_text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 10:
                st.warning("⚠️ **Teks terlalu pendek atau kosong untuk dianalisis.**")
                return
            
            st.success("✅ **Transcript berhasil diambil dan diproses.**")
            
            # Show text preview
            with st.expander("📄 **Pratinjau Teks yang Dianalisis**"):
                st.text_area(
                    "Teks yang akan dianalisis:",
                    cleaned_text[:1000] + ("..." if len(cleaned_text) > 1000 else ""),
                    height=150,
                    disabled=True
                )
                st.caption(f"Panjang total teks: {len(cleaned_text)} karakter")
            
            # Load model and predict
            model, tokenizer = load_model_and_tokenizer()
            if model is None or tokenizer is None:
                st.error("❌ **Gagal memuat model. Pastikan file model tersedia.**")
                return
            
            with st.spinner("🤖 Menganalisis dengan AI..."):
                detected_labels, all_scores = predict_hate_speech(cleaned_text, model, tokenizer)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 **Hasil Analisis**")
            display_results(detected_labels, all_scores)
            
            # Detailed scores
            with st.expander("📈 **Detail Skor Semua Kategori**"):
                col1, col2 = st.columns(2)
                
                for i, label in enumerate(LABELS):
                    score = all_scores[i]
                    emoji = LABEL_COLORS.get(label, "⚪")
                    
                    if i % 2 == 0:
                        with col1:
                            st.metric(
                                f"{emoji} {LABEL_DESCRIPTIONS[label]}", 
                                f"{score:.1f}%"
                            )
                    else:
                        with col2:
                            st.metric(
                                f"{emoji} {LABEL_DESCRIPTIONS[label]}", 
                                f"{score:.1f}%"
                            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🤖 Powered by IndoBERTweet + BiGRU • 🇮🇩 Bahasa Indonesia
    <br>
    ⚠️ Hasil analisis adalah prediksi AI dan bukan keputusan hukum final
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()