#jurnaling page
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Analisis Journaling",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("model/Pemodelan_GRU.keras")
    except:
        # Fallback for demonstration
        st.warning("Model tidak ditemukan. Menggunakan data demo.")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open("model/tokenizer.pkl", "rb") as handle:
            return pickle.load(handle)
    except:
        # Fallback for demonstration
        return None

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

MAXLEN = 100  

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        color: white;
        background-color: #1E1E5A;
        padding: 1.5rem;
        text-align: center;
        border-radius: 10px 10px 0 0;
        margin-bottom: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .sub-header {
        color: white;
        background-color: #1E1E5A;
        font-size: 1rem;
        padding: 0.7rem;
        text-align: center;
        border-radius: 0 0 10px 10px;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    .text-input-container {
        background-color: #f7f7f7;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
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
    .result-container {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin-top: 2rem;
        margin-bottom: 2rem;
        animation: fadeIn 0.5s;
    }
    .custom-button-container {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .emotion-label {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .emotion-score {
        margin-left: 10px;
        color: #555;
    }
    .chart-title {
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Analisis Jurnaling</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">TULISKAN EKSPRESIMU DENGAN KATA-KATA</div>', unsafe_allow_html=True)

# Input text container
text_input = st.text_area("", height=200, placeholder="Tuliskan isi jurnal anda di sini...")

# Centered analysis button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
    analyze_button = st.button("Analisis Teks", key="analyze", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to perform text analysis
def analyze_text(text):
    # Demonstration mode if model is not available
    if model is None or tokenizer is None:
        # Return sample predictions
        return {
            "dominant_emotion": "marah",
            "top_emotions": [
                ("marah", 0.61), 
                ("jijik", 0.34), 
                ("sedih", 0.03)
            ],
            "positive_score": 5.0,
            "negative_score": 95.0,
            "text": text
        }
    
    # Preprocessing
    def clean_text(text):
        import re
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove links
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-letter characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    clean_text_input = clean_text(text)
    
    # Tokenization & Padding
    text_seq = tokenizer.texts_to_sequences([clean_text_input])
    text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=MAXLEN, padding='post')
    
    # Prediction with GRU model
    prediction = model.predict(text_padded)
    predicted_class = np.argmax(prediction)
    
    # Emotion labels
    label_mapping = {0: "marah", 1: "sedih", 2: "jijik", 3: "takut", 4: "bahagia", 5: "netral", 6: "terkejut"}
    emotion_label = label_mapping[predicted_class]
    
    # Calculate negative & positive scores
    negative_labels = ["marah", "sedih", "jijik", "takut"]
    positive_labels = ["bahagia", "netral", "terkejut"]
    
    negative_score = sum(prediction[0][i] for i, label in label_mapping.items() if label in negative_labels) * 100
    positive_score = sum(prediction[0][i] for i, label in label_mapping.items() if label in positive_labels) * 100
    
    # Three dominant emotions
    dominant_emotions = {
        "marah": prediction[0][0], 
        "sedih": prediction[0][1], 
        "jijik": prediction[0][2],
        "takut": prediction[0][3], 
        "bahagia": prediction[0][4], 
        "netral": prediction[0][5],
        "terkejut": prediction[0][6]
    }
    top_3_emotions = sorted(dominant_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "dominant_emotion": emotion_label,
        "top_emotions": top_3_emotions,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "text": text
    }

# Run analysis when button is clicked
if analyze_button:
    if text_input:
        with st.spinner('Menganalisis teks...'):
            result = analyze_text(text_input)
            st.session_state.text_analysis_result = result
        st.success('Analisis selesai!')
        st.rerun()
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")

# Display results if available
if 'text_analysis_result' in st.session_state:
    result = st.session_state.text_analysis_result
    
    # Display detected emotions
    st.markdown("### Hasil:")
    st.markdown("#### Emosi Yang Terdeteksi:")
    
    # Display emotion results with better formatting
    for emotion, score in result["top_emotions"]:
        emotion_name = emotion.capitalize()
        score_percent = score*100
        
        # Color coding for emotions
        color = "#1E88E5"  # Default blue
        if emotion == "marah":
            color = "#E53935"  # Red for anger
        elif emotion == "sedih":
            color = "#7986CB"  # Blue-purple for sadness
        elif emotion == "jijik":
            color = "#8BC34A"  # Green for disgust
        elif emotion == "takut":
            color = "#FFB74D"  # Orange for fear
        elif emotion == "bahagia":
            color = "#4CAF50"  # Green for happiness
            
        st.markdown(
            f'<div class="emotion-label" style="color:{color};">{emotion_name} <span class="emotion-score">{score_percent:.1f}%</span></div>',
            unsafe_allow_html=True
        )
    
    # Also display happiness score separately
    st.markdown(
        f'<div class="emotion-label" style="color:#4CAF50;">Happy <span class="emotion-score">{result["positive_score"]:.1f}%</span></div>',
        unsafe_allow_html=True
    )
    
    # Create visualization for emotions
    st.markdown('<div class="chart-title">Top 3 Predictions</div>', unsafe_allow_html=True)
    
    # Create dataframe for chart
    emotions = [e[0].capitalize() for e in result["top_emotions"]]
    scores = [e[1]*100 for e in result["top_emotions"]]
    
    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Custom colors
    colors = ['#5DADE2', '#58D68D', '#F4D03F']
    
    # Create bars
    bars = ax.barh(emotions, scores, color=colors, height=0.5)
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                va='center', fontweight='bold')
    
    # Customize chart
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='-', alpha=0.2)
    
    # Display chart
    st.pyplot(fig)
    
    # Multimodal results button with standardized style
    st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
    multimodal_button = st.button("Lihat Hasil Multimodal", key="multimodal", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if multimodal_button:
        st.switch_page("pages/3_hasil.py")