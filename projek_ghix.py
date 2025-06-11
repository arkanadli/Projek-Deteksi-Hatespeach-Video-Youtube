import streamlit as st
import re
import torch
import torch.nn as nn
import numpy as np
import os
import requests
from transformers import AutoTokenizer, AutoModel
from youtube_transcript_api import YouTubeTranscriptApi
from huggingface_hub import hf_hub_download
import zipfile
import tempfile

st.set_page_config(page_title="Deteksi Hate Speech pada Video", layout="centered")
st.title("🎥 Deteksi Hate Speech dari Video YouTube")

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
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

# 📦 Download file dari Google Drive dengan requests
def download_from_gdrive(file_id, destination):
    """Download file dari Google Drive menggunakan requests"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    with requests.Session() as session:
        response = session.get(url, stream=True)
        
        # Handle Google Drive's virus scan warning
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(url, params=params, stream=True)
                break
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

# 📦 Load model dan tokenizer dengan error handling yang lebih baik
@st.cache_resource
def load_model_tokenizer():
    try:
        # Opsi 1: Coba load dari Hugging Face Hub (recommended)
        try:
            tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
            bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
            st.success("✅ Model loaded from Hugging Face Hub")
        except Exception as e:
            st.warning(f"⚠️ Could not load from HF Hub: {e}")
            
            # Opsi 2: Download dari Google Drive sebagai fallback
            tokenizer_zip_id = "14xqjAOJDw57wq9XpnYK8sWRQk-KO4uIu"
            tokenizer_dir = "indobertweet-tokenizer"
            
            if not os.path.exists(tokenizer_dir):
                st.info("📥 Downloading tokenizer from Google Drive...")
                
                # Create temp directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_path = os.path.join(temp_dir, "tokenizer.zip")
                    
                    # Download zip file
                    download_from_gdrive(tokenizer_zip_id, zip_path)
                    
                    # Extract zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tokenizer_dir)
            
            # Load tokenizer and model from local directory
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            bert = AutoModel.from_pretrained(tokenizer_dir, local_files_only=True)
            st.success("✅ Model loaded from Google Drive")

        # Download model weights
        model_file_id = "1OpDWxAl7bcKCm9OVb0vZCmEDZcBi424B"
        model_path = "model_bigru.pth"
        
        if not os.path.exists(model_path):
            st.info("📥 Downloading model weights...")
            download_from_gdrive(model_file_id, model_path)

        # Initialize model
        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please check your Google Drive file IDs and permissions")
        return None, None

# 🎯 Main app
def main():
    youtube_url = st.text_input("Masukkan URL Video YouTube:")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            
            # Load model dengan error handling
            with st.spinner("🔄 Loading model..."):
                model, tokenizer = load_model_tokenizer()
            
            if model is None or tokenizer is None:
                st.error("❌ Failed to load model. Please check the setup.")
                return

            st.info("📥 Mengambil transcript dari video...")
            try:
                # Coba bahasa Indonesia dulu, lalu English
                transcript = None
                for lang in ['id', 'en']:
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                        break
                    except:
                        continue
                
                if transcript is None:
                    # Coba auto-generated transcript
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                
                full_text = " ".join([entry['text'] for entry in transcript])
                st.success("✅ Transcript berhasil diambil!")
                
                # Show transcript preview
                with st.expander("📄 Lihat Transcript"):
                    st.text_area("Transcript:", full_text, height=200)

                # Process text
                cleaned_text = preprocessing(full_text)
                
                # Tokenize dengan max_length yang lebih kecil untuk menghindari memory issues
                inputs = tokenizer(
                    cleaned_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=128  # Reduced from 192
                )

                # Predict
                with torch.no_grad():
                    logits = model(**inputs)
                    probs = torch.sigmoid(logits)
                    predictions = (probs > 0.5).int().numpy()[0]

                # Display results
                st.subheader("📊 Hasil Deteksi Hate Speech:")
                detected = [LABELS[i] for i, val in enumerate(predictions) if val == 1]
                
                if detected:
                    st.warning("⚠️ **Hate Speech Detected:**")
                    for label in detected:
                        st.write(f"- 🚨 **{label}**")
                else:
                    st.success("✅ **Tidak terdeteksi hate speech dalam video ini.**")
                
                # Show confidence scores
                with st.expander("📈 Confidence Scores"):
                    for i, (label, prob) in enumerate(zip(LABELS, probs[0])):
                        confidence = prob.item() * 100
                        st.write(f"**{label}**: {confidence:.2f}%")
                        
            except Exception as e:
                st.error(f"❌ Gagal mengambil transcript: {str(e)}")
                st.info("💡 Tip: Pastikan video memiliki subtitle/caption yang tersedia")
        else:
            st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")

if __name__ == "__main__":
    main()