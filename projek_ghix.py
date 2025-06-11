import streamlit as st
import re
import torch
import torch.nn as nn
import numpy as np
import os
import yt_dlp # Pastikan yt-dlp terinstal (pip install yt-dlp)
import json
import tempfile
from pathlib import Path
import urllib.parse as urlparse
from urllib.parse import parse_qs
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file # Pastikan safetensors terinstal (pip install safetensors)

st.set_page_config(
    page_title="Deteksi Hate Speech pada Video",
    page_icon="🇮🇩",
    layout="centered"
)
st.title("🎥 Deteksi Hate Speech dari Video YouTube")

# --- Bagian Arsitektur Model ---
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

# --- Label Klasifikasi ---
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

# --- Fungsi Pemuatan Model dan Tokenizer (di-cache oleh Streamlit) ---
@st.cache_resource # Gunakan cache Streamlit untuk menghindari loading berulang
def load_bert_and_model(tokenizer_path, model_path, model_name="indolem/indobertweet-base-uncased",
                       hidden_size=512, num_classes=13):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Menggunakan device: {device}")
    try:
        # Menggunakan path lokal untuk tokenizer
        if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path): # Cek jika itu folder yang valid
            st.info(f"📥 Memuat tokenizer dari path lokal: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            bert = AutoModel.from_pretrained(tokenizer_path).to(device)
        else:
            st.warning(f"Folder tokenizer lokal tidak ditemukan di {tokenizer_path}. Mengunduh tokenizer online sebagai fallback.")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            bert = AutoModel.from_pretrained(model_name).to(device)
        st.success("✅ Tokenizer dan model BERT dasar berhasil dimuat!")

        # Inisialisasi arsitektur model
        model = IndoBERTweetBiGRU(
            bert=bert, # Gunakan instance bert yang sudah dimuat
            hidden_size=hidden_size,
            num_classes=num_classes
        )

        # Load state dict dari file model
        if not os.path.exists(model_path):
            st.error(f"Model file tidak ditemukan: {model_path}")
            raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")

        st.info(f"🔒 Memuat bobot model dari: {model_path}")
        try:
            # Cek apakah file .safetensors
            if model_path.endswith(".safetensors"):
                state_dict = load_file(model_path)
            else: # Anggap sebagai .pth
                state_dict = torch.load(model_path, map_location=device)

            # Handle berbagai format checkpoint (seperti di kode lokal Anda)
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                elif 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                else:
                    model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

            model.to(device) # Pindahkan model ke device yang benar
            model.eval() # Set ke evaluation mode
            st.success(f"✅ Model IndoBERTweetBiGRU berhasil dimuat dari: {model_path}!")

        except Exception as e:
            st.error(f"❌ Error loading model weights: {e}")
            st.info("💡 Tips: Pastikan parameter model_name, hidden_size, dan num_classes sesuai dengan model yang disimpan. Pastikan juga file model tidak korup.")
            raise

    except Exception as e:
        st.error(f"Error inisialisasi model/tokenizer: {e}")
        # import traceback
        # st.exception(traceback.format_exc()) # Uncomment for full traceback during debugging
        raise

    return model, tokenizer, device


# --- YouTubeTranscriptProcessor dari Kode Lokal Anda (diperbarui) ---
class YouTubeTranscriptProcessor:
    def __init__(self, model, tokenizer, device, num_classes=13):
        """
        Inisialisasi processor dengan model, tokenizer, dan device yang sudah dimuat.

        Args:
            model: Instance model IndoBERTweetBiGRU yang sudah dimuat.
            tokenizer: Instance tokenizer yang sudah dimuat.
            device: Device (torch.device) di mana model berada.
            num_classes (int): Jumlah kelas output.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_classes = num_classes

        self.label_mapping = {i: label for i, label in enumerate(LABELS)} # Gunakan LABELS global

    def extract_video_id(self, url):
        # Implementasi dari kode lokal Anda
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
            return video_id

        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url):
            return url
        return None

    def get_youtube_transcript_ytdlp(self, video_url_or_id, languages=['id', 'en']):
        # Implementasi dari kode lokal Anda
        try:
            video_id = self.extract_video_id(video_url_or_id)
            if not video_id:
                # Perbaikan kecil di sini untuk URL yang benar
                video_url = video_url_or_id if video_url_or_id.startswith('http') else f"https://www.youtube.com/watch?v={video_url_or_id}"
                video_id = video_url_or_id # fallback jika input adalah ID saja
            else:
                video_url = f"https://www.youtube.com/watch?v={video_id}"


            st.info(f"🔍 Mengambil transkrip untuk video: {video_url}")

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
                st.info("📋 Mengambil informasi video...")
                info = ydl.extract_info(video_url, download=False)

                video_id = info.get('id', video_id)
                title = info.get('title', 'Unknown Title')
                duration = info.get('duration', 0)

                st.write(f"  Judul: {title}")
                st.write(f"  Durasi: {duration} detik")
                st.write(f"  Video ID: {video_id}")

                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})

                st.write(f"  Subtitle manual: {list(subtitles.keys())}")
                st.write(f"  Subtitle otomatis: {list(automatic_captions.keys())}")

                transcript_data = None
                used_language = None
                is_automatic = False

                for lang in languages:
                    if lang in subtitles:
                        st.success(f"✓ Menggunakan subtitle manual: {lang}")
                        transcript_data = subtitles[lang]
                        used_language = lang
                        is_automatic = False
                        break

                if not transcript_data:
                    for lang in languages:
                        if lang in automatic_captions:
                            st.success(f"✓ Menggunakan subtitle otomatis: {lang}")
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
                        st.warning(f"⚠ Menggunakan bahasa yang tersedia: {lang} (Bukan pilihan utama)")

                if not transcript_data:
                    raise Exception("Tidak ada subtitle yang tersedia untuk video ini")

                st.info("📥 Mengunduh subtitle...")
                subtitle_url = None
                for format_info in transcript_data:
                    if format_info.get('ext') == 'json3':
                        subtitle_url = format_info.get('url')
                        break
                if not subtitle_url and transcript_data:
                    subtitle_url = transcript_data[0].get('url')

                if not subtitle_url:
                    raise Exception("URL subtitle tidak ditemukan")

                import urllib.request
                import json

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

                st.success(f"✓ Transkrip berhasil diambil!")
                st.write(f"  Bahasa: {used_language}")
                st.write(f"  Tipe: {'Otomatis' if is_automatic else 'Manual'}")
                st.write(f"  Segmen: {result['total_segments']}")
                st.write(f"  Kalimat: {result['total_sentences']}")
                st.write(f"  Panjang teks: {len(full_text)} karakter")

                return result

        except Exception as e:
            st.error(f"❌ Error mengambil transkrip dengan yt-dlp: {e}")
            raise Exception(f"Gagal mengambil transkrip: {e}")

    def preprocessing(self, text):
        # Implementasi dari kode lokal Anda
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

    def tokenize_text(self, text, max_length=192): # Sesuaikan max_length dengan yang digunakan model Anda
        # Implementasi dari kode lokal Anda
        try:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return encoded
        except Exception as e:
            st.error(f"✗ Error dalam tokenisasi: {e}")
            raise

    def predict_text(self, text, threshold=0.5):
        # Implementasi dari kode lokal Anda
        try:
            cleaned_text = self.preprocessing(text)
            tokens = self.tokenize_text(cleaned_text)

            # Pastikan input tensor berada di device yang sama dengan model
            tokens = {key: val.to(self.device) for key, val in tokens.items()}

            with torch.no_grad():
                logits = self.model(tokens['input_ids'], tokens['attention_mask'])
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > threshold).int()

                probs_numpy = probabilities.cpu().numpy().flatten()
                preds_numpy = predictions.cpu().numpy().flatten()

            predicted_labels = []
            label_details = {}

            for i, (prob, pred) in enumerate(zip(probs_numpy, preds_numpy)):
                label_name = self.label_mapping[i]
                label_details[label_name] = {
                    'probability': float(prob),
                    'predicted': bool(pred)
                }
                if pred == 1:
                    predicted_labels.append(label_name)

            result = {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'predicted_labels': predicted_labels,
                'binary_predictions': preds_numpy.tolist(),
                'probabilities': probs_numpy.tolist(),
                'label_details': label_details,
                'threshold_used': threshold
            }
            return result
        except Exception as e:
            st.error(f"✗ Error dalam prediksi: {e}")
            raise

    def analyze_youtube_video(self, video_url_or_id, threshold=0.5, segment_analysis=False, languages=['id', 'en']):
        # Implementasi dari kode lokal Anda
        try:
            st.info("🎬 Memulai analisis video YouTube...")
            transcript_data = self.get_youtube_transcript_ytdlp(video_url_or_id, languages)

            st.info("\n🔍 Menganalisis transkrip keseluruhan...")
            overall_result = self.predict_text(transcript_data['full_text'], threshold)

            segment_results = []
            if segment_analysis and transcript_data['segments']:
                st.info(f"\n🔍 Menganalisis {len(transcript_data['segments'])} segment...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, segment in enumerate(transcript_data['segments']):
                    if len(segment['text'].strip()) > 0:
                        segment_pred = self.predict_text(segment['text'], threshold)
                        segment_result = {
                            'segment_index': i,
                            'start_time': segment['start'],
                            'duration': segment.get('duration', 0),
                            'text': segment['text'],
                            'prediction': segment_pred
                        }
                        segment_results.append(segment_result)
                    progress_bar.progress((i + 1) / len(transcript_data['segments']))
                    status_text.text(f"Progress analisis segment: {i + 1}/{len(transcript_data['segments'])}")
                status_text.empty() # Clear the status text after completion

            final_result = {
                'video_info': {
                    'video_id': transcript_data['video_id'],
                    'video_url': transcript_data['video_url'],
                    'title': transcript_data.get('title', 'Unknown Title'),
                    'language': transcript_data['language'],
                    'language_code': transcript_data['language_code'],
                    'transcript_type': transcript_data['transcript_type'],
                    'is_generated': transcript_data['is_generated'],
                    'total_segments': transcript_data['total_segments'],
                    'total_sentences': transcript_data['total_sentences'],
                    'total_duration': transcript_data['total_duration']
                },
                'transcript': {
                    'full_text': transcript_data['full_text'],
                    'clean_sentences': transcript_data['clean_sentences'],
                    'text_length': len(transcript_data['full_text'])
                },
                'overall_analysis': overall_result,
                'segment_analysis': segment_results if segment_analysis else None,
                'summary': self.generate_summary(overall_result, segment_results if segment_analysis else None)
            }
            return final_result

        except Exception as e:
            st.error(f"❌ Error dalam analisis video: {e}")
            raise

    def generate_summary(self, overall_result, segment_results=None):
        # Implementasi dari kode lokal Anda
        summary = {
            'overall_status': 'CLEAN',
            'risk_level': 'LOW',
            'detected_categories': overall_result['predicted_labels'],
            'total_positive_labels': len(overall_result['predicted_labels']),
            'highest_probability': max(overall_result['probabilities']) if overall_result['probabilities'] else 0
        }

        if any(label.startswith('HS') for label in overall_result['predicted_labels']):
            summary['overall_status'] = 'HATE_SPEECH_DETECTED'
            if 'HS_Strong' in overall_result['predicted_labels']:
                summary['risk_level'] = 'HIGH'
            elif 'HS_Moderate' in overall_result['predicted_labels']:
                summary['risk_level'] = 'MEDIUM'
            elif 'HS_Weak' in overall_result['predicted_labels']:
                summary['risk_level'] = 'LOW-MEDIUM'

        elif 'Abusive' in overall_result['predicted_labels']:
            summary['overall_status'] = 'ABUSIVE_CONTENT'
            summary['risk_level'] = 'MEDIUM'

        elif 'PS' in overall_result['predicted_labels']:
            summary['overall_status'] = 'SEXUAL_CONTENT'
            summary['risk_level'] = 'MEDIUM'

        if segment_results:
            problematic_segments = []
            for segment in segment_results:
                if segment['prediction']['predicted_labels']:
                    problematic_segments.append({
                        'start_time': segment['start_time'],
                        'text': segment['text'][:100] + '...' if len(segment['text']) > 100 else segment['text'],
                        'labels': segment['prediction']['predicted_labels']
                    })
            summary['problematic_segments_count'] = len(problematic_segments)
            summary['problematic_segments'] = problematic_segments[:5]
        return summary

# --- Main App Logic (Streamlit) ---
def main():
    # Konfigurasi path dan parameter model (sesuaikan dengan setup Anda)
    # PASTIKAN PATH INI SESUAI DENGAN LOKASI FILE DI LINGKUNGAN DEPLOYMENT STREAMLIT ANDA
    # Jika Anda mendeploy di Streamlit Community Cloud, pastikan file-file ini di-upload
    # ke repo GitHub Anda dan pathnya relatif.
    # Contoh jika ada di folder 'models' di root repo:
    # tokenizer_dir = "models/indobertweet-base-uncased"
    # model_file_path = "models/final_model.pth"

    # Untuk Streamlit Community Cloud, umumnya direktori root adalah '/app/repo_name'
    # Jadi jika folder 'indobertweet-base-uncased' dan 'final_model.pth' ada di root repo:
    tokenizer_dir = "./indobertweet-base-uncased"
    model_file_path = "./final_model.pth" # Sesuaikan jika sudah diubah ke .safetensors

    # Inisialisasi model, tokenizer, dan device hanya sekali menggunakan cache
    model, tokenizer, device = load_bert_and_model(
        tokenizer_path=tokenizer_dir,
        model_path=model_file_path,
        model_name="indolem/indobertweet-base-uncased", # Gunakan nama Hugging Face untuk fallback
        hidden_size=512,
        num_classes=13
    )

    # Inisialisasi processor dengan objek yang sudah dimuat
    # Ini memastikan YouTubeTranscriptProcessor tidak menyimpan objek yang tidak dapat di-hash
    # dan tidak melakukan pemuatan model/tokenizer itu sendiri.
    processor = YouTubeTranscriptProcessor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_classes=13
    )

    youtube_url = st.text_input("🔗 Masukkan URL Video YouTube:")

    if youtube_url:
        video_id = processor.extract_video_id(youtube_url)
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}") # Perbaiki format URL video preview

            # Tambahkan opsi untuk threshold dan segment analysis
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Threshold Deteksi (0.0 - 1.0)", 0.0, 1.0, 0.5, 0.05)
            with col2:
                segment_analysis_choice = st.checkbox("Analisis Per Segmen", value=False)

            lang_input = st.text_input("Bahasa Subtitle (pisahkan dengan koma, cth: id,en)", value="id,en").strip()
            languages = [lang.strip() for lang in lang_input.split(',')]

            if st.button("🚀 Analisis Video"):
                st.subheader("📊 Hasil Deteksi Hate Speech:")
                try:
                    result = processor.analyze_youtube_video(
                        youtube_url,
                        threshold=threshold,
                        segment_analysis=segment_analysis_choice,
                        languages=languages
                    )

                    # Tampilkan hasil di Streamlit
                    st.success("✅ Analisis Selesai!")

                    # Ringkasan
                    summary = result['summary']
                    st.markdown("---")
                    st.markdown(f"### Ringkasan Analisis")
                    st.write(f"**Status Keseluruhan**: {summary['overall_status']}")
                    st.write(f"**Tingkat Risiko**: {summary['risk_level']}")
                    st.write(f"**Total Kategori Terdeteksi**: {summary['total_positive_labels']}/{processor.num_classes}")
                    st.write(f"**Probabilitas Tertinggi**: {summary['highest_probability']:.4f}")

                    if summary['overall_status'] != 'CLEAN':
                        st.error(f"🚨 **PERINGATAN**: {summary['overall_status']}")
                        st.write(f"**Kategori Terdeteksi**: {', '.join([LABEL_DESCRIPTIONS[l] for l in summary['detected_categories']])}")
                    else:
                        st.success("🎉 **Konten Bersih dari Ujaran Kebencian dan Konten Bermasalah**")

                    st.markdown("---")
                    st.markdown(f"### Detail Analisis Keseluruhan")
                    overall = result['overall_analysis']
                    if overall['predicted_labels']:
                        for label in overall['predicted_labels']:
                            prob_score = overall['label_details'][label]['probability'] * 100
                            st.write(f"- 🔴 **{LABEL_DESCRIPTIONS[label]}** ({prob_score:.1f}%)")
                    else:
                        st.write("Tidak ada kategori ujaran kebencian/konten bermasalah yang terdeteksi secara keseluruhan.")

                    with st.expander("📈 Detail Skor Semua Kategori"):
                        for label, details in overall['label_details'].items():
                            status_icon = "✓" if details['predicted'] else "✗"
                            st.write(f"{status_icon} **{LABEL_DESCRIPTIONS[label]}**: {details['probability']:.4f}")

                    if segment_analysis_choice and result['segment_analysis']:
                        st.markdown("---")
                        st.markdown(f"### Analisis Per Segmen")
                        problematic_count = summary.get('problematic_segments_count', 0)
                        st.write(f"**Total Segmen Bermasalah**: {problematic_count}/{result['video_info']['total_segments']}")

                        if problematic_count > 0:
                            st.info("Berikut adalah beberapa segmen yang terdeteksi bermasalah:")
                            for i, seg in enumerate(summary.get('problematic_segments', []), 1):
                                st.markdown(f"**Segmen {i}. Waktu: {seg['start_time']:.1f}s**")
                                st.write(f"Teks: `{seg['text']}`")
                                st.write(f"Label Terdeteksi: {', '.join([LABEL_DESCRIPTIONS[l] for l in seg['labels']])}")
                                st.markdown("---")
                        else:
                            st.info("Tidak ada segmen spesifik yang terdeteksi bermasalah.")

                    with st.expander("📄 Transkrip Lengkap"):
                        st.text_area("Teks Transkrip", result['transcript']['full_text'], height=300)

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat menganalisis video: {e}")
                    st.info("Pastikan:")
                    st.markdown("- Video memiliki subtitle (manual atau otomatis) dalam bahasa yang dipilih.")
                    st.markdown("- URL video YouTube valid dan bersifat publik.")
                    st.markdown("- Koneksi internet stabil.")
                    st.markdown("- `yt-dlp` dan dependensi lainnya terinstal dengan benar (`pip install yt-dlp transformers torch safetensors numpy`).")
                    import traceback
                    st.exception(traceback.format_exc()) # Menampilkan traceback penuh untuk debugging

        else:
            st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")

    st.markdown("---")
    with st.expander("ℹ️ Cara Menggunakan dan Konfigurasi Lokal"):
        st.markdown(
            """
            1.  **Siapkan File Model dan Tokenizer Lokal**:
                * Pastikan Anda memiliki folder tokenizer `indobertweet-base-uncased` dan file model `final_model.pth` (atau `final_model.safetensors` jika sudah dikonversi) di **lokasi yang dapat diakses oleh aplikasi Streamlit**.
                * **Sangat disarankan** untuk meletakkan folder `indobertweet-base-uncased` dan file `final_model.pth` (atau `.safetensors`) di **direktori yang sama** dengan file `.py` Streamlit ini, atau di sub-folder yang relevan.
                * Jika Anda menggunakan Streamlit Community Cloud (atau platform hosting lainnya), Anda harus menyertakan folder dan file ini dalam repositori GitHub Anda.
                * **Edit kode ini**: Ubah nilai `tokenizer_dir` dan `model_file_path` di awal fungsi `main()` agar sesuai dengan lokasi file Anda (misalnya: `tokenizer_dir = "my_models/indobertweet-base-uncased"`).

            2.  **Instal Dependensi**:
                ```bash
                pip install streamlit torch transformers safetensors numpy yt-dlp
                ```
                *Pastikan versi **PyTorch >= 2.6.0** (lihat panduan di bawah jika bermasalah).*

            3.  **Masukkan URL Video YouTube**: Paste URL video yang ingin Anda analisis.
            4.  **Atur Opsi Analisis**: Pilih `Threshold Deteksi` dan apakah ingin `Analisis Per Segmen`.
            5.  **Klik 'Analisis Video'**: Tunggu prosesnya selesai. Durasi analisis akan bervariasi tergantung panjang transkrip.
            """
        )

    st.markdown("---")
    with st.expander("🔧 Panduan Konversi Manual ke SafeTensors (Direkomendasikan!)"):
        st.markdown(
            """
            Jika Anda memiliki model PyTorch dalam format `.pth` dan ingin mengonversinya ke SafeTensors untuk keamanan dan efisiensi yang lebih baik:

            ```python
            # Jalankan di lingkungan dengan PyTorch >= 2.6 dan safetensors terinstal
            from safetensors.torch import save_file
            import torch
            import os

            # Ganti dengan path lengkap ke file .pth Anda
            pth_file_path = "path/to/your/final_model.pth"
            safetensors_output_path = "path/to/save/final_model.safetensors"

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
                    print(f"Sekarang Anda dapat mengganti `model_file_path` di kode Streamlit dengan path ke file .safetensors ini.")
                except Exception as e:
                    print(f"❌ Gagal mengonversi model: {e}")
                    print("Pastikan file .pth tidak korup dan versi PyTorch Anda terbaru (>= 2.6 direkomendasikan).")
            ```

            **Langkah-langkah:**
            1.  Pastikan Anda memiliki PyTorch versi **2.6 atau lebih baru**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
            2.  Instal pustaka SafeTensors: `pip install safetensors`
            3.  Unduh file `.pth` Anda secara manual ke komputer lokal Anda.
            4.  Buat dan jalankan skrip Python di atas (ganti `pth_file_path` dan `safetensors_output_path` sesuai).
            5.  Perbarui `model_file_path` di kode Streamlit Anda dengan path ke file `.safetensors` yang baru.
            """
        )


if __name__ == "__main__":
    main()