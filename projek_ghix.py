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

st.set_page_config(
    page_title="Deteksi Hate Speech pada Video",
    page_icon="🇮🇩",
    layout="centered"
)
st.title("🎥 Deteksi Hate Speech dan Abusive Language dari Video YouTube Berbahasa Indonesia")

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
    "HS_Weak", "HS_Moderate", "HS_Strong", "PS" # PS = Konten Positif
]

LABEL_DESCRIPTIONS = {
    "HS": "Ujaran Kebencian",
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
    "PS": "Ujaran Positif"
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
def get_transcript_from_searchapi(video_id: str, api_key: str) -> Optional[Dict]:
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
        return data
    except Exception as e:
        st.error(f"Gagal mengambil transcript: {str(e)}")
        return None

# 📦 Download model dan tokenizer
@st.cache_resource
def load_model_tokenizer():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased").to(device)

        model_safetensors_id = "1SfyGkTgRxjx3JEwZ79zJuz5wciOH6d6_"
        safetensors_path = "final_model.safetensors"
        safetensors_url = f"https://drive.google.com/uc?id={model_safetensors_id}"

        if not os.path.exists(safetensors_path):
            try:
                gdown.download(safetensors_url, safetensors_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Gagal download model SafeTensors: {str(e)}")
                st.info("💡 Pastikan file dapat diakses publik dan ID benar.")
                return None, None, None

        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)

        if os.path.exists(safetensors_path):
            try:
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict)
                model.to(device)
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

# Fungsi untuk memprediksi satu kalimat
def predict_sentence(text, model, tokenizer, device, threshold=0.5):
    cleaned_text = preprocessing(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).int().numpy()[0]

    detected_labels = [LABELS[i] for i, val in enumerate(predictions) if val == 1]

    # Detail probabilitas untuk setiap label
    label_probs = {LABELS[i]: float(probs[0][i]) for i in range(len(LABELS))}

    return detected_labels, label_probs


# 🎯 Main App
def main():
    # SearchAPI key disematkan langsung di sini
    api_key = "Quq35w6JgdtV1fJcnACFK4qF"

    with st.expander("ℹ️ Cara Menggunakan"):
        st.markdown(
            """
            1. **Paste URL video YouTube** yang ingin dianalisis.
            2. **Pastikan video memiliki subtitle bahasa Indonesia**.
            3. **Klik tombol 'Analisis Video'** dan tunggu prosesnya selesai.

            **Catatan**:
            Proses analisis membutuhkan waktu beberapa detik tergantung panjang video dan jumlah kalimat.
            """
        )

    youtube_url = st.text_input("🔗 Masukkan URL Video YouTube:")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            with st.spinner("📦 Loading model dan tokenizer..."):
                model, tokenizer, device = load_model_tokenizer()

            if model is None or tokenizer is None or device is None:
                st.error("❌ Gagal memuat model. Periksa koneksi dan file model.")
                return

            if st.button("🚀 Analisis Video", use_container_width=True):
                with st.spinner("📥 Mengambil transcript dari video..."):
                    transcript_data = get_transcript_from_searchapi(video_id, api_key)

                if not transcript_data or "transcripts" not in transcript_data:
                    st.error("❌ Gagal mengambil transcript. Pastikan video memiliki subtitle bahasa Indonesia.")
                    return

                transcript_entries = transcript_data["transcripts"]
                available_languages = transcript_data.get("available_languages", [])
                is_not_auto_generated = any(lang["name"] == "Indonesian"for lang in available_languages)

                if not is_not_auto_generated:
                    st.warning("⚠️ Transkrip ini merupakan Auto-Generated dan mungkin mengandung kesalahan.")

                full_text = " ".join([entry['text'] + ". " for entry in transcript_entries]) # Menambahkan titik di setiap entry
                st.success("✅ Transcript berhasil diambil! ")

                with st.expander("📄 Cuplikan Transcript"):
                    st.text_area("", full_text[:1000] + ("..." if len(full_text) > 1000 else ""), height=200)

                st.subheader("🔍 Menganalisis Konten Video per Kalimat...")

                sentences = re.split(r'(?<=[.!?])\s+|\n', full_text)
                clean_sentences = [s.strip() for s in sentences if s.strip()]

                if not clean_sentences:
                    st.warning("Tidak ada kalimat yang dapat dianalisis dari transkrip ini.")
                    return

                problematic_sentences_count = 0
                problematic_sentences_details = []

                progress_text = "Analisis kalimat sedang berjalan. Mohon tunggu..."
                my_bar = st.progress(0, text=progress_text)

                for i, sentence in enumerate(clean_sentences):
                    detected_labels, label_probs = predict_sentence(sentence, model, tokenizer, device)

                    # Logika untuk kalimat "bermasalah": setiap label selain 'PS'
                    is_problematic = any(label != "PS" for label in detected_labels)

                    if is_problematic:
                        problematic_sentences_count += 1
                        problematic_sentences_details.append({
                            "kalimat": sentence,
                            "label_terdeteksi": [LABEL_DESCRIPTIONS[label] for label in detected_labels if label != "PS"], # Hanya tampilkan label selain PS
                            "probabilitas": {LABEL_DESCRIPTIONS[k]: f"{v:.1%}" for k, v in label_probs.items()}
                        })

                    progress_percentage = (i + 1) / len(clean_sentences)
                    my_bar.progress(progress_percentage, text=f"{progress_text} {int(progress_percentage * 100)}%")

                my_bar.empty()

                st.subheader("📊 Ringkasan Hasil Deteksi Hate Speech:")

                total_sentences = len(clean_sentences)
                if total_sentences > 0:
                    percentage_problematic = (problematic_sentences_count / total_sentences) * 100
                    # Tambahan Warning jika lebih dari 50% kalimat tidak positif
                    if percentage_problematic > 10.0:
                        st.error("⚠️ **PERINGATAN**: Lebih dari 10% konten video ini terdeteksi sebagai **konten tidak positif** (ujaran kebencian/abusive) dan **tidak layak dikonsumsi** secara umum.")
                    st.error(f"Dari **{total_sentences} kalimat**, **{problematic_sentences_count} kalimat ({percentage_problematic:.1f}%)** terklasifikasi sebagai **konten bermasalah** (Ujaran Kebencian / Abusive).")
                else:
                    st.warning("Tidak ada kalimat untuk dianalisis.")

                if problematic_sentences_details:
                    st.info("🚨 Berikut adalah kalimat-kalimat yang terdeteksi bermasalah:")
                    for idx, detail in enumerate(problematic_sentences_details, 1):
                        st.markdown(f"---")
                        st.markdown(f"**Kalimat {idx}:** {detail['kalimat']}")
                        # Pastikan hanya menampilkan label yang terdeteksi dan bukan PS
                        display_labels = [label for label in detail['label_terdeteksi'] if label != "Konten Positif"]
                        if display_labels:
                            st.markdown(f"**Label Terdeteksi:** {', '.join(display_labels)}")
                        else:
                            st.markdown(f"**Label Terdeteksi:** (Tidak ada label spesifik yang terdeteksi selain 'Konten Positif')") # Fallback jika hanya PS yang terdeteksi tapi diabaikan

                        with st.expander("Detail Probabilitas:"):
                            for label_desc, prob in detail['probabilitas'].items():
                                st.write(f"- **{label_desc}**: {prob}")
                else:
                    st.success("✅ Tidak terdeteksi adanya hate speech atau konten bermasalah dalam transkrip ini.")

        else:
            st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")

if __name__ == "__main__":
    main()