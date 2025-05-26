import streamlit as st
import re

st.set_page_config(page_title="Deteksi Hate Speech pada Video", layout="centered")

st.title("🎥 Deteksi Hate Speech dari Video YouTube")

# Form input URL
youtube_url = st.text_input("Masukkan URL Video YouTube:")

def extract_video_id(url):
    # Ekstraksi ID video dari berbagai format URL YouTube
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

if youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        # Tampilkan preview video YouTube
        st.video(f"https://www.youtube.com/watch?v={video_id}")

        # Placeholder hasil deteksi
        st.markdown("### 🔍 Hasil Deteksi Hate Speech:")
        st.write("🚧 Model belum terkoneksi. Hasil akan muncul di sini setelah integrasi selesai.")
    else:
        st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")
