import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown
import os
import base64
from PIL import Image
import io
import cv2

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
    st.session_state.current_frame = None
    st.session_state.webcam_running = False
    st.session_state.start_time = time.time()

# Create a callback for receiving webcam frames
def handle_webcam_frame(image_data):
    if st.session_state.is_analyzing and image_data is not None:
        st.session_state.current_frame = image_data
        # Set a flag to trigger rerun
        st.session_state.webcam_update = True
        st.rerun()

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

# Process base64 image from webcam
def process_webcam_frame(base64_image):
    try:
        # Remove data URL prefix
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB numpy array
        frame_rgb = np.array(image)
        
        # Predict expression
        if model_loaded:
            pred_idx, pred_prob = predict_expression(frame_rgb, model_effnet)
            
            current_time = time.time()
            elapsed_time = current_time - st.session_state.start_time
            
            st.session_state.predictions.append(pred_idx)
            st.session_state.prediction_timestamps.append(elapsed_time)
            
            current_expression = categories[pred_idx]
            current_accuracy = f"{pred_prob[0][pred_idx]*100:.2f}%"
            
            return frame_rgb, current_expression, current_accuracy
        
        return frame_rgb, "Model not loaded", "0%"
    except Exception as e:
        st.error(f"Error processing webcam frame: {e}")
        return None, "Error", "0%"

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

# Create columns for metrics
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

# Create container for webcam display
frame_placeholder = st.empty()

# Custom webcam component using streamlit-webrtc
if model_loaded:
    # Create a custom HTML component for webcam access
    webcam_html = """
    <div style="display: flex; flex-direction: column; align-items: center;">
        <video id="webcam" autoplay playsinline style="width: 640px; height: 480px; border-radius: 10px;"></video>
        <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
    </div>
    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let stream = null;
        let captureInterval = null;
        let isCapturing = false;

        // Function to initialize webcam
        async function setupWebcam() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 },
                    audio: false
                });
                video.srcObject = stream;
                return true;
            } catch (err) {
                console.error("Error accessing webcam:", err);
                return false;
            }
        }

        // Function to capture frame
        function captureFrame() {
            if (!stream) return;
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to Streamlit using the Streamlit component callback
            if (window.parent && window.parent.Streamlit) {
                window.parent.Streamlit.setComponentValue(imageData);
            }
        }

        // Setup webcam and start capturing when component loads
        setupWebcam().then(success => {
            if (success) {
                isCapturing = true;
                captureInterval = setInterval(captureFrame, 500); // Capture every 500ms
            }
        });

        // Clean up when component is unmounted
        window.addEventListener('beforeunload', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            if (captureInterval) {
                clearInterval(captureInterval);
            }
        });
    </script>
    """
    
    # Display the webcam component if analyzing
    if st.session_state.is_analyzing:
        frame_component = st.components.v1.html(webcam_html, height=500, key="webcam_component")
        st.session_state.webcam_running = True
    else:
        # Display a placeholder when not analyzing
        if not st.session_state.results_ready:
            frame_placeholder.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; 
                        width: 640px; height: 480px; background-color: #111c4e; 
                        border-radius: 10px; margin: 0 auto;">
                <p style="color: white; font-size: 16px;">Kamera tidak aktif</p>
            </div>
            """, unsafe_allow_html=True)

# Start/Stop Analysis Button
button_col = st.columns([1, 2, 1])[1]
with button_col:
    st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
    if st.button('Mulai Analisis' if not st.session_state.is_analyzing else 'Akhiri Analisis'):
        if not st.session_state.is_analyzing:
            # Start new analysis
            st.session_state.predictions = []
            st.session_state.prediction_timestamps = []
            st.session_state.start_time = time.time()
            st.session_state.analysis_start_time = time.time()
            st.session_state.last_capture_time = 0
            st.session_state.results_ready = False
            st.session_state.current_expression = "-"
            st.session_state.current_accuracy = "-"
            st.session_state.video_started = False
            st.session_state.is_analyzing = True
            
            # Show stimulus video after delay
            if not st.session_state.video_started and (time.time() - st.session_state.analysis_start_time) >= 10:
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
                """, unsafe_allow_html=True)
                st.session_state.video_started = True
        else:
            # End analysis and show results
            st.session_state.is_analyzing = False
            st.session_state.results_ready = True
            st.session_state.webcam_running = False
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
    st.markdown('</div>', unsafe_allow_html=True)

# Process webcam frame if available
if 'current_frame' in st.session_state and st.session_state.current_frame is not None and st.session_state.is_analyzing:
    frame_rgb, current_expression, current_accuracy = process_webcam_frame(st.session_state.current_frame)
    
    if frame_rgb is not None:
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        expression_placeholder.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Ekspresi Terdeteksi</div>
            <div class="metric-value" id="expression-value">{current_expression}</div>
        </div>
        """, unsafe_allow_html=True)
        
        accuracy_placeholder.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Akurasi</div>
            <div class="metric-value" id="accuracy-value">{current_accuracy}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Reset the flag
        st.session_state.current_frame = None

# Display results
if st.session_state.results_ready and st.session_state.predictions:
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
