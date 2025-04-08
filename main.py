#main page
import streamlit as st

def set_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
            font-family: Arial, sans-serif;
            margin: 0; /* Menghilangkan margin default body */
            padding: 0; /* Menghilangkan padding default body */
        }

        .title {
            color: #f4f4f7;
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #f4f4f7;
            margin-bottom: 20px;
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
            width: 50%;
        }

        .stButton > button:hover {
            background-color: #0056b3;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .center-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Aplikasi Pendeteksi Depresi", layout="wide")
    set_custom_css()
    
    st.markdown('<div class="title">Selamat Datang di Aplikasi Pendeteksi Indikasi Depresi</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Aplikasi ini menganalisis ekspresi wajah dan teks untuk mendeteksi potensi indikasi depresi.</div>', unsafe_allow_html=True)
    
    # Membuat container untuk tombol dan memusatkannya
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    if st.button("Mulai Analisis"):
        st.switch_page("pages/1_analisis_wajah.py")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()