import streamlit as st
import numpy as np
import io
import tempfile
import os
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent
import noisereduce as nr
from scipy.signal import butter, lfilter
import plotly.graph_objects as go
import plotly.express as px

# Налаштування сторінки
st.set_page_config(
    page_title="Аудіо Рекордер та Редактор v1.0.4",
    page_icon="🎵",
    layout="wide"
)

# Функція для обрізки тиші
def trim_silence(audio, silence_thresh=-40, min_silence_len=100):
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent:
        return audio
    start_trim = nonsilent[0][0]
    end_trim = nonsilent[-1][1]
    return audio[start_trim:end_trim]

# Функція для створення еквалайзера
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_equalizer(samples, fs, gains):
    # 5 смуг: 60, 250, 1000, 4000, 16000 Гц
    bands = [(20, 120), (120, 500), (500, 2000), (2000, 8000), (8000, 20000)]
    out = np.zeros_like(samples)
    for i, (low, high) in enumerate(bands):
        b, a = butter_bandpass(low, high, fs)
        filtered = lfilter(b, a, samples)
        gain = 10 ** (gains[i] / 20)
        out += filtered * gain
    # Нормалізація після еквалайзера
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out)) * 0.95
    return out

# Головний заголовок
st.title("🎵 Аудіо Рекордер та Редактор v1.0.4")
st.markdown("---")

# Ініціалізація стану сесії
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'processed_audio' not in st.session_state:
    st.session_state.processed_audio = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Бокова панель для еквалайзера
with st.sidebar:
    st.header("🎛️ Еквалайзер")
    
    # Слайдери для 5 смуг
    eq_gains = []
    eq_freqs = [60, 250, 1000, 4000, 16000]
    
    for i, freq in enumerate(eq_freqs):
        gain = st.slider(f'{freq} Гц', -12, 12, 0, 1, key=f'eq_{i}')
        eq_gains.append(gain)
    
    st.markdown("---")
    
    # Кнопки управління
    st.header("🎮 Управління")
    
    if st.button("📥 Завантажити аудіо", use_container_width=True):
        st.info("Використовуйте завантаження файлу нижче")
    
    if st.button("🔄 Скасувати", use_container_width=True):
        if st.session_state.history:
            st.session_state.audio_data = st.session_state.history.pop()
            st.rerun()
    
    if st.button("💾 Зберегти", use_container_width=True):
        if st.session_state.audio_data is not None:
            st.success("Використовуйте завантаження файлу нижче для збереження")

# Основна область
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Візуалізація аудіо")
    
    # Завантаження файлу
    uploaded_file = st.file_uploader(
        "Виберіть аудіо файл", 
        type=['wav', 'mp3', 'flac', 'ogg'],
        help="Підтримуються формати: WAV, MP3, FLAC, OGG"
    )
    
    if uploaded_file is not None:
        try:
            # Завантаження аудіо
            audio_bytes = uploaded_file.read()
            
            # Створення тимчасового файлу
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            # Завантаження аудіо через pydub
            audio = AudioSegment.from_file(tmp_file_path)
            
            # Конвертація в моно
            if audio.channels == 2:
                audio = audio.set_channels(1)
            
            # Отримання зразків
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples = samples / np.max(np.abs(samples))  # Нормалізація
            
            st.session_state.audio_data = {
                'samples': samples,
                'fs': audio.frame_rate,
                'duration': len(audio) / 1000.0,
                'filename': uploaded_file.name
            }
            
            # Очищення тимчасового файлу
            os.unlink(tmp_file_path)
            
            st.success(f"✅ Файл завантажено: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"❌ Помилка завантаження: {str(e)}")
    
    # Відображення графіка
    if st.session_state.audio_data is not None:
        samples = st.session_state.audio_data['samples']
        fs = st.session_state.audio_data['fs']
        duration = st.session_state.audio_data['duration']
        
        # Створення часової осі
        t = np.linspace(0, duration, len(samples))
        
        # Графік аудіо
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, 
            y=samples, 
            mode='lines',
            name='Аудіо сигнал',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title="Хвиля аудіо",
            xaxis_title="Час (секунди)",
            yaxis_title="Амплітуда",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Інформація про файл
        st.info(f"📊 Тривалість: {duration:.2f} сек | Частота: {fs} Гц | Зразків: {len(samples)}")

with col2:
    st.header("🎵 Відтворення")
    
    if st.session_state.audio_data is not None:
        # Кнопки відтворення
        col_play1, col_play2 = st.columns(2)
        
        with col_play1:
            if st.button("▶️ Відтворити", use_container_width=True):
                st.audio(uploaded_file, format='audio/wav')
        
        with col_play2:
            if st.button("🎛️ З EQ", use_container_width=True):
                # Застосування еквалайзера
                samples = st.session_state.audio_data['samples']
                fs = st.session_state.audio_data['fs']
                
                eq_samples = apply_equalizer(samples, fs, eq_gains)
                
                # Конвертація назад в аудіо
                eq_audio = (eq_samples * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    eq_audio.tobytes(),
                    frame_rate=fs,
                    sample_width=2,
                    channels=1
                )
                
                # Збереження в тимчасовий файл
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    audio_segment.export(tmp_file.name, format="wav")
                    with open(tmp_file.name, "rb") as f:
                        st.audio(f.read(), format='audio/wav')
                    os.unlink(tmp_file.name)
        
        st.markdown("---")
        
        # Обробка аудіо
        st.header("🔧 Обробка")
        
        if st.button("🧹 Обробити аудіо", use_container_width=True):
            with st.spinner("Обробка аудіо..."):
                # Збереження в історію
                st.session_state.history.append(st.session_state.audio_data.copy())
                
                samples = st.session_state.audio_data['samples']
                fs = st.session_state.audio_data['fs']
                
                # 1. Обрізка тиші
                audio_segment = AudioSegment(
                    (samples * 32767).astype(np.int16).tobytes(),
                    frame_rate=fs,
                    sample_width=2,
                    channels=1
                )
                
                trimmed = trim_silence(audio_segment, silence_thresh=audio_segment.dBFS-16, min_silence_len=100)
                proc_samples = np.array(trimmed.get_array_of_samples()).astype(np.float32)
                proc_samples = proc_samples / np.max(np.abs(proc_samples))
                
                # 2. Видалення шуму
                reduced = nr.reduce_noise(y=proc_samples, sr=fs)
                
                # 3. Нормалізація
                normed = effects.normalize(AudioSegment(
                    (reduced * 32767).astype(np.int16).tobytes(),
                    frame_rate=fs,
                    sample_width=2,
                    channels=1
                ))
                
                # 4. Еквалайзер
                eq_samples = np.array(normed.get_array_of_samples()).astype(np.float32)
                eq_samples = eq_samples / np.max(np.abs(eq_samples))
                eq_samples = apply_equalizer(eq_samples, fs, eq_gains)
                
                # Оновлення даних
                st.session_state.audio_data['samples'] = eq_samples
                st.session_state.processed_audio = eq_samples
                
                st.success("✅ Аудіо оброблено!")
                st.rerun()
        
        # Завантаження обробленого файлу
        if st.session_state.processed_audio is not None:
            st.markdown("---")
            st.header("💾 Збереження")
            
            # Конвертація в аудіо сегмент
            processed_samples = st.session_state.processed_audio
            fs = st.session_state.audio_data['fs']
            
            audio_segment = AudioSegment(
                (processed_samples * 32767).astype(np.int16).tobytes(),
                frame_rate=fs,
                sample_width=2,
                channels=1
            )
            
            # Збереження в буфер
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="wav")
            buffer.seek(0)
            
            st.download_button(
                label="📥 Завантажити оброблене аудіо",
                data=buffer.getvalue(),
                file_name="processed_audio.wav",
                mime="audio/wav",
                use_container_width=True
            )
    
    else:
        st.info("📁 Завантажте аудіо файл для початку роботи")

# Підвал
st.markdown("---")
st.markdown("🎵 **Аудіо Рекордер та Редактор v1.0.4** - Створено з Streamlit")
st.markdown("💡 **Функції**: Обрізка тиші, видалення шуму, еквалайзер, нормалізація")
st.markdown("🔧 **Нові функції v1.0.4**: Покращено шумоподавлення, додано кнопку 'Відтворити з EQ'")
