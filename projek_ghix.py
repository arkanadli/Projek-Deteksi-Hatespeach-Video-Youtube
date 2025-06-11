import streamlit as st
import torch
import torch.nn as nn
import re
import os
import requests
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel

# Set Streamlit config
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian Indonesia",
    page_icon="🇮🇩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Konstanta label
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

# Model class - Fixed version with your architecture
class IndoBERTweetBiGRU(nn.Module):
    def __init__(self, model_name="indolem/indobertweet-base-uncased", hidden_size=512, num_classes=13):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
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
        cls_output = sequence_output[:, 0, :]
        gru_output, _ = self.gru(sequence_output)
        pooled_output = gru_output.mean(dim=1)
        combined = torch.cat((pooled_output, cls_output), dim=1)
        logits = self.fc(combined)
        return logits

# Fungsi preprocessing
def preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Ambil video ID dari URL
def extract_video_id(url: str) -> Optional[str]:
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

# Ambil transcript dari SearchAPI.io
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

# Load model dan tokenizer - Fixed with proper error handling
@st.cache_resource
def load_model_tokenizer() -> Tuple[Optional[nn.Module], Optional[AutoTokenizer]]:
    try:
        model_path = "model_bigru.pth"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan: {model_path}")
            return None, None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        
        # Initialize model with correct parameters
        model = IndoBERTweetBiGRU(
            model_name="indolem/indobertweet-base-uncased",
            hidden_size=512,  # Sesuaikan dengan model Anda
            num_classes=13
        )
        
        # Try loading with weights_only=True first (safer)
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as safe_load_error:
            st.warning("Loading dengan weights_only=True gagal, mencoba metode alternatif...")
            try:
                # Fallback to regular loading (less safe but might work)
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
            except Exception as fallback_error:
                st.error(f"Gagal memuat model: {fallback_error}")
                st.info("Solusi: Upgrade PyTorch ke versi ≥2.6 atau simpan model dalam format safetensors")
                return None, None
        
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Gagal memuat model/tokenizer: {e}")
        st.info("Pastikan PyTorch versi ≥2.6 dan file model tersedia")
        return None, None

# Prediksi - Updated max_length to 192
def predict(text: str, model: nn.Module, tokenizer: AutoTokenizer) -> Tuple[List[str], List[float]]:
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=192  # Updated to match your specification
    )
    
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits)[0].numpy()
        predictions = (probs > 0.5).astype(int)
    
    labels = [LABELS[i] for i, p in enumerate(predictions) if p == 1]
    scores = [float(prob * 100) for prob in probs]
    return labels, scores

# Streamlit app
def main():
    st.title("🇮🇩 Deteksi Ujaran Kebencian dari Video YouTube (ID)")
    st.markdown("Masukkan URL video YouTube yang memiliki subtitle Bahasa Indonesia")
    
    # API Key input - Fixed the default value issue
    api_key = st.secrets.get("searchapi_key")
    if not api_key:
        api_key = st.text_input(
            "🔑 SearchAPI Key", 
            placeholder="Masukkan API key SearchAPI.io Anda",
            type="password"
        )
    
    url = st.text_input("🔗 URL Video YouTube")

    if url and api_key:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("URL YouTube tidak valid")
            return

        st.video(f"https://www.youtube.com/watch?v={video_id}")

        if st.button("🚀 Analisis Video"):
            with st.spinner("Mengambil transcript..."):
                transcript = get_transcript_from_searchapi(video_id, api_key)
            
            if not transcript:
                st.error("Tidak berhasil mengambil transcript.")
                st.info("Pastikan video memiliki subtitle bahasa Indonesia")
                return

            # Process transcript
            text = " ".join([t["text"] for t in transcript])
            cleaned = preprocessing(text)
            st.success("Transcript berhasil diambil dan dibersihkan")

            with st.expander("📄 Pratinjau Teks"):
                st.text_area("", cleaned[:1000] + ("..." if len(cleaned) > 1000 else ""), height=150)

            # Load model
            with st.spinner("Memuat model..."):
                model, tokenizer = load_model_tokenizer()
            
            if model is None or tokenizer is None:
                st.error("Gagal memuat model. Pastikan file model tersedia dan PyTorch versi terbaru.")
                return

            # Predict
            with st.spinner("Menganalisis teks..."):
                labels, scores = predict(cleaned, model, tokenizer)
            
            # Display results
            if labels:
                st.error("🚨 Terdeteksi ujaran kebencian!")
                for lbl in labels:
                    score_idx = LABELS.index(lbl)
                    st.write(f"**{LABEL_DESCRIPTIONS[lbl]}** — {scores[score_idx]:.1f}%")
            else:
                st.success("✅ Tidak terdeteksi ujaran kebencian")
            
            # Show all scores
            with st.expander("📊 Detail Skor Semua Kategori"):
                for i, (label, desc) in enumerate(zip(LABELS, [LABEL_DESCRIPTIONS[l] for l in LABELS])):
                    st.write(f"**{desc}**: {scores[i]:.1f}%")

    elif url and not api_key:
        st.warning("Masukkan API key SearchAPI.io untuk melanjutkan")
    
    # Instructions
    with st.expander("ℹ️ Cara Menggunakan"):
        st.markdown("""
        1. Dapatkan API key dari [SearchAPI.io](https://www.searchapi.io/)
        2. Masukkan API key dan URL video YouTube
        3. Pastikan video memiliki subtitle bahasa Indonesia
        4. Klik tombol 'Analisis Video'
        """)

if __name__ == "__main__":
    main()