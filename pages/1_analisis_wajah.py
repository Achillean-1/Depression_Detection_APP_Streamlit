import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.markdown("""
<style>
    .main { background-color: #0b0f2e; color: white; }
    .stApp { background-color: #0b0f2e; }
    .css-1d391kg, .css-12oz5g7 { background-color: #111c4e; border-radius: 10px; padding: 20px; }
    .stButton > button {
        background-color: #007bff;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 1rem;
        border: none;
        cursor: pointer;
        transition: 0.3s;
        display: block;
        margin: 0 auto;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: white; text-align: center; }
    .metric-container { background-color: #1a2352; padding: 10px; border-radius: 5px; text-align: center; }
    .metric-label { font-size: 14px; opacity: 0.8; }
    .metric-value { font-size: 18px; font-weight: bold; }
    .results-container { background-color: #111c4e; border-radius: 10px; padding: 20px; margin-top: 20px; }
    .custom-button-container { margin-top: 20px; margin-bottom: 20px; }
    .emotion-score { font-size: 24px; font-weight: bold; text-align: center; margin: 10px 0; }
    .negative-score { color: #ff6b6b; }
    .positive-score { color: #69db7c; }
    .stApp .stImage {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    .youtube-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 20px 0;
    }
    .youtube-embed {
        margin: 0 auto;
    }
    .hidden {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_and_load_model():
    os.makedirs('model', exist_ok=True)
    
    model_path = "model/Model_EfficientNet.keras"
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading model from Google Drive...'):
            file_id = '1yZfLsjnaXm_S8mUJ8uQkHbkVPLtNGwC0'
            
            url = f'https://drive.google.com/uc?id={file_id}'
            
            gdown.download(url, model_path, quiet=False)
    
    return load_model(model_path)

try:
    with st.spinner('Loading model...'):
        model_effnet = download_and_load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

categories = ['Angry', 'Sad', 'Happy', 'Fearful', 'Disgust', 'Neutral', 'Surprised']
positive_emotions = ['Happy', 'Surprised', 'Neutral']
negative_emotions = ['Angry', 'Sad', 'Fearful', 'Disgust']

# Frame interval for prediction (in seconds)
FRAME_INTERVAL = 0.5

def predict_expression(image, model):
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    image_array = preprocess_input(image_array)
    
    pred = model.predict(image_array)
    return np.argmax(pred), pred

def calculate_emotion_scores(predictions):
    emotion_indices = [categories[idx] for idx in predictions]

    total_predictions = len(predictions)
    negative_count = sum(1 for emotion in emotion_indices if emotion in negative_emotions)
    positive_count = sum(1 for emotion in emotion_indices if emotion in positive_emotions)
    
    negative_score = (negative_count / total_predictions) * 100 if total_predictions > 0 else 0
    positive_score = (positive_count / total_predictions) * 100 if total_predictions > 0 else 0
    
    return negative_score, positive_score

# Video processor for WebRTC
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.predictions = []
        self.prediction_timestamps = []
        self.start_time = time.time()
        self.last_capture_time = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Proses frame untuk prediksi setiap interval tertentu
        if current_time - self.last_capture_time >= FRAME_INTERVAL:
            pred_idx, pred_prob = predict_expression(img, self.model)
            
            self.predictions.append(pred_idx)
            self.prediction_timestamps.append(elapsed_time)
            
            # Perbarui status sesi
            st.session_state.predictions = self.predictions
            st.session_state.prediction_timestamps = self.prediction_timestamps
            st.session_state.current_expression = categories[pred_idx]
            st.session_state.current_accuracy = f"{pred_prob[0][pred_idx]*100:.2f}%"
            
            self.last_capture_time = current_time
        
        return av.VideoFrame.from_ndarray(img, format="rgb24")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
    st.session_state.prediction_timestamps = []
    st.session_state.is_analyzing = False
    st.session_state.results_ready = False
    st.session_state.current_expression = "-"
    st.session_state.current_accuracy = "-"
    st.session_state.video_started = False
    st.session_state.analysis_start_time = None
    st.session_state.last_capture_time = 0
    st.session_state.webrtc_context = None

st.markdown("<h1>Analisis Emosi Wajah</h1>", unsafe_allow_html=True)

youtube_placeholder = st.empty()
youtube_placeholder.markdown("""
<div class="youtube-container hidden" id="youtube-container">
    <div class="youtube-embed">
        <iframe 
            width="640" 
            height="360" 
            src="https://www.youtube.com/embed/3XA0bB79oGc?autoplay=0&mute=1" 
            frameborder="0" 
            allowfullscreen
            id="youtube-iframe">
        </iframe>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
expression_placeholder = col1.empty()
accuracy_placeholder = col2.empty()

expression_placeholder.markdown("""
<div class="metric-container">
    <div class="metric-label">Ekspresi Terdeteksi</div>
    <div class="metric-value" id="expression-value">-</div>
</div>
""", unsafe_allow_html=True)

accuracy_placeholder.markdown("""
<div class="metric-container">
    <div class="metric-label">Akurasi</div>
    <div class="metric-value" id="accuracy-value">-</div>
</div>
""", unsafe_allow_html=True)

button_col = st.columns([1, 2, 1])[1]
with button_col:
    st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
    start_stop_button = st.button(
        'Mulai Analisis' if not st.session_state.is_analyzing else 'Akhiri Analisis',
        key='start_stop_analysis_button'
    )
    st.markdown('</div>', unsafe_allow_html=True)

if start_stop_button:
    st.session_state.is_analyzing = not st.session_state.is_analyzing
    if st.session_state.is_analyzing:
        # Reset state for new analysis
        st.session_state.predictions = []
        st.session_state.prediction_timestamps = []
        st.session_state.start_time = time.time()
        st.session_state.analysis_start_time = time.time()
        st.session_state.last_capture_time = 0
        st.session_state.results_ready = False
        st.session_state.current_expression = "-"
        st.session_state.current_accuracy = "-"
        st.session_state.video_started = False
    else:
        # End analysis and show results
        st.session_state.results_ready = True
        st.session_state.current_expression = "-"
        st.session_state.current_accuracy = "-"
        youtube_placeholder.markdown("""
        <div class="youtube-container hidden" id="youtube-container">
            <div class="youtube-embed">
                <iframe 
                    width="640" 
                    height="360" 
                    src="https://www.youtube.com/embed/3XA0bB79oGc?autoplay=0&mute=1" 
                    frameborder="0" 
                    allowfullscreen
                    id="youtube-iframe">
                </iframe>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.rerun()

# WebRTC Camera Stream for Emotion Analysis
if model_loaded and st.session_state.is_analyzing:
    # Configure RTC for WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Display WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=lambda: EmotionProcessor(model_effnet),
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.session_state.webrtc_context = webrtc_ctx
    
    # Update the UI with detection results
    if webrtc_ctx.state.playing:
        # Check if video should start
        current_time = time.time()
        if not st.session_state.video_started and (current_time - st.session_state.analysis_start_time) >= 10:
            youtube_placeholder.markdown("""
            <div class="youtube-container" id="youtube-container">
                <div class="youtube-embed">
                    <iframe 
                        width="640" 
                        height="360" 
                        src="https://www.youtube.com/embed/3XA0bB79oGc?autoplay=1&mute=1" 
                        frameborder="0" 
                        allowfullscreen
                        id="youtube-iframe">
                    </iframe>
                </div>
            </div>
            <script>
                // JavaScript to ensure video starts
                document.getElementById('youtube-iframe').src = 
                    "https://www.youtube.com/embed/3XA0bB79oGc?autoplay=1&mute=1";
            </script>
            """, unsafe_allow_html=True)
            st.session_state.video_started = True
        
        # Create a container for showing live metrics
        status_container = st.container()
        
        # Use empty placeholders to update metrics
        with status_container:
            while st.session_state.is_analyzing:
                expression_placeholder.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Ekspresi Terdeteksi</div>
                    <div class="metric-value" id="expression-value">{st.session_state.current_expression}</div>
                </div>
                """, unsafe_allow_html=True)
                
                accuracy_placeholder.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Akurasi</div>
                    <div class="metric-value" id="accuracy-value">{st.session_state.current_accuracy}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add delay to reduce UI updates
                time.sleep(0.5)

# Display results after analysis is complete
if not st.session_state.is_analyzing and st.session_state.predictions:
    st.markdown("<div class='results-container'>", unsafe_allow_html=True)
    st.success("Analisis ekspresi wajah selesai!")

    negative_score, positive_score = calculate_emotion_scores(st.session_state.predictions)
    
    st.session_state.negative_score = negative_score
    st.session_state.positive_score = positive_score
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-container negative-score">
            <div class="metric-label">Skor Emosi Negatif</div>
            <div class="emotion-score negative-score">{negative_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container positive-score">
            <div class="metric-label">Skor Emosi Positif</div>
            <div class="emotion-score positive-score">{positive_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    emotion_counts = Counter(st.session_state.predictions)
    most_common_emotions = emotion_counts.most_common(3)
    total_predictions = sum(emotion_counts.values())

    st.write("**3 Emosi Paling Dominan:**")
    for emotion, count in most_common_emotions:
        percentage = (count / total_predictions) * 100
        st.write(f"- {categories[emotion]}: {percentage:.2f}%")

    if st.session_state.predictions:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f4f4f7')
        ax.set_facecolor('#f4f4f7')
        
        # Create x and y data from the predictions
        timestamps = st.session_state.prediction_timestamps
        expressions = [categories[idx] for idx in st.session_state.predictions]
        
        ax.plot(timestamps, expressions, marker='o', linestyle='-', color='#3a7aff')
        ax.set_xlabel("Waktu (detik)", color='#111c4e')
        ax.set_ylabel("Ekspresi", color='#111c4e')
        ax.set_title("Perubahan Ekspresi Selama Sesi", color='#111c4e')
        ax.tick_params(axis='x', colors='#111c4e')
        ax.tick_params(axis='y', colors='#111c4e')
        ax.spines['bottom'].set_color('#111c4e')
        ax.spines['top'].set_color('#111c4e')
        ax.spines['left'].set_color('#111c4e')
        ax.spines['right'].set_color('#111c4e')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data ekspresi yang dikumpulkan.")

    if len(st.session_state.predictions) > 1:
        change_times = []
        for i in range(1, len(st.session_state.predictions)):
            if st.session_state.predictions[i] != st.session_state.predictions[i-1]:
                change_time = st.session_state.prediction_timestamps[i]
                from_emotion = categories[st.session_state.predictions[i-1]]
                to_emotion = categories[st.session_state.predictions[i]]
                change_times.append((from_emotion, to_emotion, change_time))
        
        if change_times:
            change_times = sorted(change_times, key=lambda x: x[2])[:3]
            st.write("**3 Perubahan Ekspresi Tercepat:**")
            for from_emotion, to_emotion, change_time in change_times:
                st.write(f"- {from_emotion} → {to_emotion} pada {change_time:.2f} detik")
        else:
            st.info("Tidak ada perubahan ekspresi terdeteksi selama sesi.")

    st.session_state.emotion_data = list(zip(
        st.session_state.prediction_timestamps,
        [categories[idx] for idx in st.session_state.predictions]
    ))
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
        if st.button("Reset Analisis", key='reset_analysis_button'):
            st.session_state.predictions = []
            st.session_state.prediction_timestamps = []
            st.session_state.is_analyzing = False
            st.session_state.results_ready = False
            st.session_state.current_expression = "-"
            st.session_state.current_accuracy = "-"
            st.session_state.video_started = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
        if st.button("Lanjut ke Analisis Teks", key='next_page_to_text_analysis'):
            st.switch_page("pages/2_jurnaling.py")
        st.markdown('</div>', unsafe_allow_html=True)
