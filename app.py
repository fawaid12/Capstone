import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import re
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi stopword remover dan stopword tambahan
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

# Stopword dari Sastrawi
sastrawi_stopwords = set(factory.get_stop_words())

# Tambahan stopword informal/slang yang umum di komentar
custom_stopwords = {
    'gak', 'ga', 'aja', 'ya', 'nih', 'sih', 'nya', 'deh', 'loh', 'pak', 'apa', 'terus', 'buat', 
    'dong', 'kok', 'kan', 'tau', 'punya', 'lu', 'gue', 'udah', 'mau', 'yg', 'malah', 'orang', 'lebih',
    'cuma', 'biar', 'bgt', 'ampun', 'banget', 'mas', 'mbak', 'min',
    'lah', 'the'
}

# Gabungkan semuanya
stopwords_all = sastrawi_stopwords.union(custom_stopwords)


# --- Fungsi Utility ---

def load_data():
    # Load semua file CSV
    df_pos = pd.read_csv('Komentar_Positif.csv')
    df_neg = pd.read_csv('Komentar_Negatif.csv')
    df_neu = pd.read_csv('Komentar_Netral.csv')

    # Load topik
    df_topik1 = pd.read_csv('Topik_1_ Error-Aplikasi.csv')
    df_topik2 = pd.read_csv('Topik_2_ Login-NPWP-Daftar.csv')
    df_topik3 = pd.read_csv('Topik_3_ Lapor_Pajak-SPT.csv')
    df_topik4 = pd.read_csv('Topik_4_ OTP-Verifikasi_Email.csv')
    df_topik5 = pd.read_csv('Topik_5_ Pembayaran-Kantor_Pajak.csv')

    # Tambahkan kolom topik untuk data topik
    df_topik1['topik'] = 'Error-Aplikasi'
    df_topik2['topik'] = 'Login-NPWP-Daftar'
    df_topik3['topik'] = 'Lapor_Pajak-SPT'
    df_topik4['topik'] = 'OTP-Verifikasi_Email'
    df_topik5['topik'] = 'Pembayaran-Kantor_Pajak'

    # Gabungkan semua komentar sentimen
    df_sentimen = pd.concat([df_pos, df_neg, df_neu], ignore_index=True)

    # Gabungkan semua topik
    df_all_topik = pd.concat([df_topik1, df_topik2, df_topik3, df_topik4, df_topik5], ignore_index=True)

    return df_sentimen, df_all_topik


def preprocess_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        filtered_words = [w for w in words if w not in stopwords_all]
        return ' '.join(filtered_words)
    except Exception as e:
        st.error(f"Error di preprocess_text: {e}")
        return ""


def get_wordcloud(data, title):
    text = ' '.join(data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    st.pyplot(plt)

def plot_bar(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=data.index, y=data.values, palette='viridis', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)


# --- Load Model & Vectorizer ---

@st.cache_resource
def load_model():
    with open('svm_sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Mengubah fungsi classify_sentiment untuk menerima vectorizer
def classify_sentiment(text, model, vectorizer):
    text_proc = preprocess_text(text)
    text_vect = vectorizer.transform([text_proc])
    pred = model.predict(text_vect)[0]
    return pred

def classify_topic(text, topik_keywords):
    text_proc = preprocess_text(text)
    scores = {}
    for topik, keywords in topik_keywords.items():
        scores[topik] = sum(text_proc.count(k) for k in keywords)
    topik_terpilih = max(scores, key=scores.get)
    if scores[topik_terpilih] == 0:
        return "Tidak Teridentifikasi"
    return topik_terpilih

def get_top_words(data, n=10):
    all_words = ' '.join(data).split()
    counter = Counter(all_words)
    return counter.most_common(n)

def plot_top_words(top_words, title):
    if not top_words:
        st.warning("Tidak ada kata yang ditemukan")
        return
    
    words, counts = zip(*top_words)
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title(title)
    plt.xlabel('Frekuensi')
    plt.ylabel('Kata')
    plt.tight_layout()
    st.pyplot(plt)

# --- Main App ---
def main():
    st.set_page_config(page_title="Analisis Sentimen & Topik Komentar Aplikasi Pajak", layout="wide")
    st.title("📊 Analisis Sentimen dan Topik Komentar Aplikasi Pajak")

    # Load data, model, dan vectorizer
    df_sentimen, df_topik = load_data()
    model = load_model()
    vectorizer = load_vectorizer()

    # Sidebar dengan judul aplikasi
    st.sidebar.title("📡 GovGauge.ID")
    st.sidebar.markdown("**Analisis Layanan Pajak Digital & Manual**")

    # Sidebar menu
    menu = st.sidebar.selectbox("Pilih Menu", ["Overview", "Visualisasi Sentimen", "Visualisasi Topik"])

    st.sidebar.markdown("### Dibuat oleh:")
    st.sidebar.markdown("Capstone-LAI25-SM091")

    # Preprocessing untuk visualisasi
    df_sentimen['cleaned'] = df_sentimen['content'].astype(str).apply(preprocess_text)
    df_topik['cleaned'] = df_topik['content'].astype(str).apply(preprocess_text)

    # Topik keywords
    topik_keywords = {
        'Error-Aplikasi': ['error', 'bug', 'gagal', 'tidak', 'bisa', 'crash', 'macet', 'lambat'],
        'Login-NPWP-Daftar': ['login', 'npwp', 'daftar', 'akun', 'masuk', 'registrasi'],
        'Lapor_Pajak-SPT': ['lapor', 'pajak', 'spt', 'form', 'laporan'],
        'OTP-Verifikasi_Email': ['otp', 'verifikasi', 'email', 'kode', 'sms'],
        'Pembayaran-Kantor_Pajak': ['bayar', 'pembayaran', 'kantor', 'pajak', 'transfer', 'online']
    }

    if menu == "Overview":
        st.header("📈 Overview Analisis Sentimen dan Topik")
        st.markdown("""
        Analisis ini bertujuan untuk memahami persepsi pengguna terhadap aplikasi pajak, mengidentifikasi masalah, dan meningkatkan kualitas layanan.
        """)

        st.metric("Total Komentar", len(df_sentimen))

        st.subheader("Distribusi Komentar Berdasarkan Sentimen")
        sentimen_counts = df_sentimen['sentiment'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%',
                startangle=140, colors=['#2ecc71', '#e74c3c', '#95a5a6'])
        ax1.axis('equal')
        st.pyplot(fig1)

        st.subheader("Distribusi Komentar Berdasarkan Topik")
        topik_counts = df_topik['topik'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=topik_counts.index, y=topik_counts.values, palette='magma', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        st.pyplot(fig2)

    elif menu == "Visualisasi Sentimen":
        sentimen_option = st.sidebar.selectbox("Pilih Sentimen", ["Semua", "positive", "negative", "neutral"])
        st.header("📊 Visualisasi Berdasarkan Sentimen")

        if sentimen_option == "Semua":
            st.subheader("Jumlah Komentar per Sentimen")
            plot_bar(df_sentimen['sentiment'].value_counts(), "Jumlah Komentar per Sentimen", "Sentimen", "Jumlah Komentar")

            for sent in ['positive', 'negative', 'neutral']:
                st.subheader(f"Wordcloud Komentar {sent.capitalize()}")
                data_sent = df_sentimen[df_sentimen['sentiment'] == sent]['cleaned']
                get_wordcloud(data_sent, f"Wordcloud Komentar {sent.capitalize()}")

                st.markdown(f"**Top Kata - {sent.capitalize()}**")
                top_words = get_top_words(data_sent)
                plot_top_words(top_words, f"Top 10 Kata pada Sentimen {sent}")
        else:
            st.subheader(f"Visualisasi Sentimen: {sentimen_option.capitalize()}")
            data_sent = df_sentimen[df_sentimen['sentiment'] == sentimen_option]['cleaned']
            get_wordcloud(data_sent, f"Wordcloud Komentar {sentimen_option.capitalize()}")

            top_words = get_top_words(data_sent)
            plot_top_words(top_words, f"Top 10 Kata pada Sentimen {sentimen_option}")

    elif menu == "Visualisasi Topik":
        topik_list = ["Semua"] + sorted(df_topik['topik'].unique())
        topik_option = st.sidebar.selectbox("Pilih Topik", topik_list)
        st.header("📊 Visualisasi Berdasarkan Topik")

        if topik_option == "Semua":
            st.subheader("Jumlah Komentar per Topik")
            topik_counts = df_topik['topik'].value_counts()
            plot_bar(topik_counts, "Jumlah Komentar per Topik", "Topik", "Jumlah Komentar")

            for topik in topik_counts.index:
                st.subheader(f"Wordcloud Komentar: {topik}")
                data_topik = df_topik[df_topik['topik'] == topik]['cleaned']
                get_wordcloud(data_topik, f"Wordcloud Komentar {topik}")

                st.markdown(f"**Top Kata - {topik}**")
                top_words = get_top_words(data_topik)
                plot_top_words(top_words, f"Top 10 Kata pada Topik {topik}")
        else:
            st.subheader(f"Visualisasi Topik: {topik_option}")
            data_topik = df_topik[df_topik['topik'] == topik_option]['cleaned']
            get_wordcloud(data_topik, f"Wordcloud Komentar {topik_option}")

            top_words = get_top_words(data_topik)
            plot_top_words(top_words, f"Top 10 Kata pada Topik {topik_option}")

if __name__ == "__main__":
    main()
