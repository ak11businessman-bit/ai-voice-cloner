import streamlit as st
import torch
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import split_on_silence
from f5_tts.api import F5TTS
from gtts import gTTS
import os
import tempfile

# إعداد الصفحة
st.set_page_config(page_title="Pro Voice Cloner", layout="centered")

# تصميم الصفحة
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1 { color: #38bdf8; text-align: center; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ Pro Voice Cloner Studio")
st.write("استنسخ أي صوت بجودة عالية. ارفع العينة، اكتب النص، واستمتع.")

# تحميل الموديل (باستخدام cache_resource عشان يحمله مرة واحدة)
@st.cache_resource
def load_models():
    with st.spinner("جاري تحميل الموديل... استنى شوية (أول مرة بس)"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f5tts = F5TTS(device=device)
    return f5tts

# تحميل الموديل
f5tts = load_models()

def remove_silence_from_audio(audio_path, min_silence_len=200, silence_thresh=-40):
    audio = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=100)
    if len(chunks) > 0:
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        combined.export(audio_path, format="wav")
    return audio_path

with st.sidebar:
    st.header("Settings")
    language = st.radio("اللغة", ["English (USA)", "Arabic"])
    speed = st.slider("السرعة", 0.4, 1.2, 0.65)
    consent = st.checkbox("أوافق على أنني أملك هذا الصوت أو لدي إذن باستخدامه")

uploaded_audio = st.file_uploader("ارفع العينة (10-15 ثانية)", type=["wav", "mp3"])
text_input = st.text_area("النص", placeholder="اكتب النص هنا...")

if st.button("🎧 توليد الصوت", type="primary"):
    if not consent:
        st.error("⚠️ You must agree to the terms first.")
    elif not text_input or not uploaded_audio:
        st.error("Error: Please provide text and audio.")
    else:
        with st.spinner("جاري التوليد... استنى شوية"):
            try:
                # حفظ الصوت المرفوع
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    tmp_audio.write(uploaded_audio.read())
                    speaker_audio = tmp_audio.name

                # تنظيف الصوت
                audio = AudioSegment.from_file(speaker_audio)
                clean_audio = normalize(audio)
                clean_audio_path = "clean_ref.wav"
                clean_audio.export(clean_audio_path, format="wav")

                output_path = "output.wav"

                if language == "English (USA)":
                    f5tts.infer(
                        ref_file=clean_audio_path,
                        ref_text="", 
                        gen_text=text_input,
                        file_wave=output_path,
                        speed=speed,
                        remove_silence=True,
                        nfe_step=64,
                        cfg_strength=2.5
                    )
                else:
                    tts_ar = gTTS(text=text_input, lang='ar', slow=False)
                    tts_ar.save("ar_temp.mp3")
                    ar_audio = AudioSegment.from_file("ar_temp.mp3")
                    ar_audio.export(output_path, format="wav")

                remove_silence_from_audio(output_path, min_silence_len=250, silence_thresh=-45)

                generated_audio = AudioSegment.from_file(output_path)
                final_audio = generated_audio.low_pass_filter(3000) 
                final_audio = normalize(final_audio)
                final_audio.export(output_path, format="wav")

                st.success("✅ Success! High Quality Generated.")
                st.audio(output_path, format="audio/wav")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
