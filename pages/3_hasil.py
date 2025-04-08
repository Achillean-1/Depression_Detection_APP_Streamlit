#hasil page
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Multimodal", layout="wide")

# Custom CSS
st.markdown("""
<style>
.section-header {
    color: #f4f4f7;
    font-weight: bold;
    margin-bottom: 20px;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
            
}
.positive-score {
    color: #2ecc71;
    font-weight: bold;
}
            
.negative-score {
    color: #e74c3c;
    font-weight: bold;
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
    width: 100%; /* Full width within container */
}

.stButton > button:hover {
    background-color: #0056b3;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.custom-button-container {
    margin-top: 20px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<h1 class="section-header">📊 Laporan Analisis Multimodal</h1>', unsafe_allow_html=True)

# Pastikan data dari sesi tersedia
if 'emotion_data' in st.session_state and 'text_analysis_result' in st.session_state:
    # Data Analisis Wajah
    face_emotion_data = st.session_state.emotion_data
    
    # Data Analisis Teks
    text_result = st.session_state.text_analysis_result
    
    # Hitung dominan emosi wajah
    dominant_face_emotions = {}
    for _, emotion in face_emotion_data:
        dominant_face_emotions[emotion] = dominant_face_emotions.get(emotion, 0) + 1
    
    top_face_emotions = sorted(dominant_face_emotions.items(), key=lambda x: x[1], reverse=True)
    
    # Mendapatkan top emosi dari teks
    text_emotions = {emotion: score for emotion, score in text_result.get("top_emotions", [])}
    
    # Menggabungkan emosi dari wajah dan teks
    combined_emotions = {}
    
    # Normalisasi skor emosi wajah
    total_face_frames = len(face_emotion_data)
    for emotion, count in dominant_face_emotions.items():
        normalized_score = count / total_face_frames
        combined_emotions[emotion] = combined_emotions.get(emotion, 0) + normalized_score
    
    # Tambahkan skor emosi teks (assuming scores are already normalized between 0 and 1)
    for emotion, score in text_emotions.items():
        combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score
    
    # Rata-rata skor (dibagi 2 karena ada 2 sumber)
    for emotion in combined_emotions:
        combined_emotions[emotion] /= 2
    
    top_combined_emotions = sorted(combined_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Analisis Perubahan Emosi
    emotion_changes = {}
    for i in range(1, len(face_emotion_data)):
        if face_emotion_data[i][1] != face_emotion_data[i-1][1]:
            key = f"{face_emotion_data[i-1][1]} → {face_emotion_data[i][1]}"
            time_diff = face_emotion_data[i][0] - face_emotion_data[i-1][0]
            emotion_changes[key] = time_diff
    
    # Get face emotion scores
    if 'negative_score' in st.session_state and 'positive_score' in st.session_state:
        positive_face_score = st.session_state.positive_score
        negative_face_score = st.session_state.negative_score
    else:
        # Fallback kalau tidak tersedia
        positive_emotions = ['Happy', 'Surprised', 'Neutral']
        negative_emotions = ['Angry', 'Sad', 'Fearful', 'Disgust']
        
        positive_face_count = sum(dominant_face_emotions.get(emotion, 0) for emotion in positive_emotions)
        negative_face_count = sum(dominant_face_emotions.get(emotion, 0) for emotion in negative_emotions)
        total_emotions = positive_face_count + negative_face_count
        
        if total_emotions > 0:
            positive_face_score = (positive_face_count / total_emotions) * 100
            negative_face_score = (negative_face_count / total_emotions) * 100
        else:
            positive_face_score = negative_face_score = 0
    
    # Get text emotion scores
    positive_text_score = text_result['positive_score']
    negative_text_score = text_result['negative_score']
    
    # Calculate average positive and negative scores
    avg_positive_score = (positive_face_score + positive_text_score) / 2
    avg_negative_score = (negative_face_score + negative_text_score) / 2
    
    # Summary Section
    st.markdown('<h2 class="section-header">📝 Ringkasan Umum</h2>', unsafe_allow_html=True)
    
    # Tampilkan ringkasan
    st.write(f"**Durasi Analisis:** {face_emotion_data[-1][0]:.2f} detik")
    st.write(f"**Jumlah Perubahan Emosi:** {len(emotion_changes)}")
    
    # Display average emotional scores
    st.markdown("### Skor Rata-rata Emosi (Wajah & Teks)")
    st.write(f"🟢 Rata-rata Emosi Positif: <span class='positive-score'>{avg_positive_score:.1f}%</span>", unsafe_allow_html=True)
    st.write(f"🔴 Rata-rata Emosi Negatif: <span class='negative-score'>{avg_negative_score:.1f}%</span>", unsafe_allow_html=True)
    
    # Create a horizontal bar chart for average scores
    fig, ax = plt.subplots(figsize=(10, 3))
    labels = ['Positif', 'Negatif']
    values = [avg_positive_score, avg_negative_score]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.barh(labels, values, color=colors, height=0.5)
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                va='center', fontweight='bold')
    
    # Customize chart
    ax.set_xlim(0, 100)
    ax.set_xlabel('Persentase (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='-', alpha=0.2)
    
    # Display chart
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("### Top 3 Emosi Gabungan (Wajah & Teks)")
    for emotion, score in top_combined_emotions:
        st.write(f"- {emotion.capitalize()}: {score*100:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analisis Wajah Section
    st.markdown('<h2 class="section-header">😀 Analisis Ekspresi Wajah</h2>', unsafe_allow_html=True)
    
    # Top 3 emosi wajah
    st.write("### Top 3 Emosi Wajah")
    for emotion, count in top_face_emotions[:3]:
        percentage = (count / total_face_frames) * 100
        st.write(f"- {emotion.capitalize()}: {percentage:.1f}% ({count} kali)")
    
    st.write(f"### Skor Emosi Wajah")
    st.write(f"🟢 Skor Positif: <span class='positive-score'>{positive_face_score:.1f}%</span>", unsafe_allow_html=True)
    st.write(f"🔴 Skor Negatif: <span class='negative-score'>{negative_face_score:.1f}%</span>", unsafe_allow_html=True)
    
    # Grafik Perubahan Emosi
    fig, ax = plt.subplots(figsize=(12, 6))
    timestamps = [data[0] for data in face_emotion_data]
    emotions = [data[1] for data in face_emotion_data]
    
    # Get unique emotions for better plotting
    unique_emotions = list(set(emotions))
    emotion_to_num = {emotion: i for i, emotion in enumerate(unique_emotions)}
    emotion_nums = [emotion_to_num[emotion] for emotion in emotions]
    
    ax.plot(timestamps, emotion_nums, marker='o')
    ax.set_xlabel('Waktu (detik)')
    ax.set_ylabel('Emosi')
    ax.set_yticks(range(len(unique_emotions)))
    ax.set_yticklabels(unique_emotions)
    ax.set_title('Perubahan Emosi Selama Analisis')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Perubahan Emosi Tercepat
    if emotion_changes:
        st.write("### Perubahan Emosi Tercepat")
        for change, time in sorted(emotion_changes.items(), key=lambda x: x[1])[:3]:
            st.write(f"- {change} dalam {time:.2f} detik")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analisis Teks Section
    st.markdown('<h2 class="section-header">📝 Analisis Teks Jurnaling</h2>', unsafe_allow_html=True)
    
    # Tampilkan top 3 emosi dari teks
    st.write("### Top 3 Emosi Terdeteksi")
    for emotion, score in text_result["top_emotions"][:3]:
        st.write(f"- {emotion.capitalize()}: {score*100:.1f}%")
    
    st.write(f"### Skor Emosi")
    st.write(f"🟢 Skor Positif: <span class='positive-score'>{positive_text_score:.1f}%</span>", unsafe_allow_html=True)
    st.write(f"🔴 Skor Negatif: <span class='negative-score'>{negative_text_score:.1f}%</span>", unsafe_allow_html=True)
    
    # Simpan teks asli
    st.write("### Teks Jurnal")
    st.write(text_result['text'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Kesimpulan Akhir
    st.markdown('<h2 class="section-header">🔍 Kesimpulan Akhir</h2>', unsafe_allow_html=True)
    
    # Use average scores for the final conclusion
    if avg_negative_score > avg_positive_score:
        st.error("⚠️ Berdasarkan analisis multimodal, Anda mungkin memerlukan dukungan emosional.")
        st.write("Disarankan untuk berbicara dengan konselor atau psikolog.")
    else:
        st.success("✅ Kondisi emosi Anda relatif stabil.")
        st.write("Tetap jaga kesehatan mental dan lanjutkan kegiatan positif.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    st.warning("Silakan lakukan analisis wajah dan jurnaling terlebih dahulu.")

if st.button("Ulangi Analisis"):
    st.session_state.clear()
    st.switch_page("pages/1_analisis_wajah.py")

st.markdown('</div>', unsafe_allow_html=True)