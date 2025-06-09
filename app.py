import streamlit as st
st.set_page_config(page_title="Analisis Sentimen & Topik Terhadap Pelayanan Pajak", layout="wide")
# Tambah Font Awesome untuk ikon
st.markdown('<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">', unsafe_allow_html=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud
import pickle
import re
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from PIL import Image


# Inisialisasi stopword remover dan stopword tambahan
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

# Stopword dari Sastrawi
sastrawi_stopwords = set(factory.get_stop_words())

# Tambahan stopword informal/slang yang umum di komentar
custom_stopwords = {
    'gak', 'ga', 'aja', 'ya', 'nih', 'sih', 'nya', 'deh', 'loh', 'pak', 'apa', 'terus', 'buat', 
    'dong', 'kok', 'kan', 'tau', 'punya', 'lu', 'gue', 'udah', 'mau', 'yg', 'malah', 'orang', 'lebih',
    'cuma', 'biar', 'bgt', 'ampun', 'banget', 'mas', 'mbak', 'min', 'sm', 'b', 'g', 'klo', 'dah', 'tp', 
    'lah', 'the', 'trus', 'pagi', 'udh', 'kak', 'ni', 'gmn', 'tuh', 'tgl', 'blm', 'gk', 'sy', 'tdk',
    'sdh', 'tpi', 'utk', 'd', 'wong', 'dr', 'ngga', 'org', 'dr', 'tlp', 'paje', 'tlong', 'koc', 'lgi', 'lg',
    'klu', 'skli', 'gw', 'jg', 'eh', 'jd', 'kek', 'tak'
}

# Gabungkan semuanya
stopwords_all = sastrawi_stopwords.union(custom_stopwords)

# --- Fungsi Utility Tambahan ---
def show_score_cards(title, data_dict, icons_dict, warna_icon="#2e7d32", warna_bg="#e8f5e9"):
    st.subheader(title)
    cols = st.columns(len(data_dict))
    for i, (label, value) in enumerate(data_dict.items()):
        with cols[i]:
            st.markdown(f'''
                <div style="background-color:{warna_bg}; padding:15px; border-radius:10px; text-align:center;">
                    <i class="{icons_dict.get(label, 'fa-solid fa-circle-info')}" style="font-size:30px;color:{warna_icon};"></i><br>
                    <strong style="font-size:16px;">{label}</strong><br>
                    <span style="font-size:24px;color:{warna_icon};">{value:,}</span>
                </div>
            ''', unsafe_allow_html=True)

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


def get_wordcloud(data, title, max_words=20):
    text = ' '.join(data)
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    st.pyplot(plt)

def plot_bar(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=data.index, y=data.values, palette='Paired', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
def get_popular_comments(text_series, top_n=5):
    return text_series.value_counts().head(top_n).reset_index().rename(columns={'index': 'Komentar', text_series.name: 'Jumlah'})



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

def get_top_words_list(data, n=5):
    # Pastikan semua data berupa string
    all_words = ' '.join(data.astype(str)).split()
    counter = Counter(all_words)
    df = pd.DataFrame(counter.most_common(n), columns=['Kata', 'Jumlah'])
    return df


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
    
    
    # Load data, model, dan vectorizer
    df_sentimen, df_topik = load_data()
    model = load_model()
    vectorizer = load_vectorizer()

    #logo taxgaugeid
    st.sidebar.image("Logo Tax Gauge ID.png", use_container_width=True)

    # Sidebar dengan judul aplikasi
    st.sidebar.markdown("**Sentimen Jadi Ukuran, Layanan Jadi Tujuan**")

    # Sidebar menu
    menu = st.sidebar.selectbox("Pilih Menu", ["Overview", "Visualisasi Sentimen", "Visualisasi Topik"])

    # Tambahkan selectbox tambahan berdasarkan menu
    selected_sentimen = None
    selected_topik = None

    if menu == "Visualisasi Sentimen":
        selected_sentimen = st.sidebar.selectbox("Pilih Sentimen", ["Semua", "positive", "negative", "neutral"])
    elif menu == "Visualisasi Topik":
        topik_options = [
            "Semua",
            "Error-Aplikasi",
            "Login-NPWP-Daftar",
            "Lapor_Pajak-SPT",
            "OTP-Verifikasi_Email",
            "Pembayaran-Kantor_Pajak"
        ]
        selected_topik = st.sidebar.selectbox("Pilih Topik", topik_options)
        
    # Info pembuat
    st.sidebar.markdown("---")
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
        st.title("📊 Aplikasi Analisis Sentimen Publik terhadap Layanan Pajak di Indonesia")
        st.header("📈 Overview Analisis Sentimen dan Topik")
        st.markdown("""
        Analisis sentimen dan topik komentar terhadap kualitas layanan pajak untuk mengidentifikasi masalah utama dan meningkatkan kualitas layanan.
        """)

        total_komentar = len(df_sentimen)
        st.metric("Total Komentar", total_komentar)

        # Tambahkan di bagian Overview
        jumlah_per_sentimen = df_sentimen['sentiment'].value_counts().to_dict()
        ikon_sentimen = {
            'positive': 'fa-solid fa-face-smile',
            'negative': 'fa-solid fa-face-frown',
            'neutral': 'fa-solid fa-face-meh'
        }
        show_score_cards("Jumlah Komentar per Sentimen", jumlah_per_sentimen, ikon_sentimen)
        
        jumlah_per_topik = df_topik['topik'].value_counts().to_dict()
        ikon_topik = {
            'Error-Aplikasi': 'fa-solid fa-bug',
            'Login-NPWP-Daftar': 'fa-solid fa-id-card',
            'Lapor_Pajak-SPT': 'fa-solid fa-file-invoice-dollar',
            'OTP-Verifikasi_Email': 'fa-solid fa-envelope-open-text',
            'Pembayaran-Kantor_Pajak': 'fa-solid fa-building-columns'
        }
        show_score_cards("Jumlah Komentar per Topik", jumlah_per_topik, ikon_topik, warna_icon="#6a1b9a", warna_bg="#f3e5f5")

        sentimen_counts = df_sentimen['sentiment'].value_counts()
        st.subheader("Distribusi Komentar Berdasarkan Sentimen")
        
        # Ambil jumlah label dari sentimen_counts
        num_labels = len(sentimen_counts)
        
        # Ambil warna dari colormap magma dan ubah ke format hex
        colors = [mcolors.to_hex(c) for c in cm.Paired(np.linspace(0.2, 0.8, num_labels))]
        
        # Buat pie chart
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%',
                startangle=140, colors=colors)
        ax1.axis('equal')
        st.pyplot(fig1)

        topik_counts = df_topik['topik'].value_counts()
        st.subheader("Distribusi Komentar Berdasarkan Topik")
        
        fig2, ax2 = plt.subplots(figsize=(8,4))
        sns.barplot(x=topik_counts.index, y=topik_counts.values, palette='Paired', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)

    # --- Menu: Visualisasi Sentimen ---
    elif menu == "Visualisasi Sentimen":
        st.header("📊 Visualisasi Berdasarkan Sentimen")
    
        if selected_sentimen and selected_sentimen != "Semua":
            df_filtered = df_sentimen[df_sentimen['sentiment'] == selected_sentimen]
        else:
            df_filtered = df_sentimen
    
        sentimen_counts = df_filtered['sentiment'].value_counts()
    
        st.subheader("📌 Ringkasan Komentar per Sentimen")
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Negatif", sentimen_counts.get('negative', 0))
        col2.metric("Netral", sentimen_counts.get('neutral', 0))
        col3.metric("Positif", sentimen_counts.get('positive', 0))

    
        if selected_sentimen == "Semua":
            for sent in ['positive', 'negative', 'neutral']:
                st.subheader(f"Wordcloud Komentar {sent.capitalize()}")
                data_sent = df_sentimen[df_sentimen['sentiment'] == sent]['cleaned']
                if not data_sent.empty and data_sent.str.strip().any():
                    get_wordcloud(data_sent, f"Wordcloud Komentar {sent.capitalize()}")
                    top_words = get_top_words(data_sent)
                    plot_top_words(top_words, f"Top 10 Kata pada Sentimen {sent}")
                else:
                    st.warning(f"Tidak ada data komentar untuk sentimen **{sent}**.")
        else:
            st.subheader(f"Visualisasi untuk Sentimen: {selected_sentimen.capitalize()}")
            data_sent = df_filtered['cleaned']
            if not data_sent.empty and data_sent.str.strip().any():
                get_wordcloud(data_sent, f"Wordcloud Komentar {selected_sentimen.capitalize()}")
                top_words = get_top_words(data_sent)
                plot_top_words(top_words, f"Top 10 Kata pada Sentimen {selected_sentimen.capitalize()}")
            else:
                st.warning(f"Tidak ada data komentar untuk sentimen **{selected_sentimen}**.")


    elif menu == "Visualisasi Topik":
        st.header("📊 Visualisasi Berdasarkan Topik")

        if selected_topik and selected_topik != "Semua":
            df_filtered = df_topik[df_topik['topik'] == selected_topik]
        else:
            df_filtered = df_topik

        topik_counts = df_filtered['topik'].value_counts()

        st.subheader("📌 Ringkasan Komentar per Topik")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Error-Aplikasi", topik_counts.get('Error-Aplikasi', 0))
        col2.metric("Login-NPWP-Daftar", topik_counts.get('Login-NPWP-Daftar', 0))
        col3.metric("Lapor_Pajak-SPT", topik_counts.get('Lapor_Pajak-SPT', 0))
        col4.metric("OTP-Verifikasi_Email", topik_counts.get('OTP-Verifikasi_Email', 0))
        col5.metric("Pembayaran-Kantor_Pajak", topik_counts.get('Pembayaran-Kantor_Pajak', 0))

        if selected_topik == "Semua":
            tabel_top_words = []

            for topik in topik_counts.index:
                data_topik = df_filtered[df_filtered['topik'] == topik]['cleaned'].dropna()
                
                if data_topik.empty:
                    continue
                
                top_words_df = get_top_words_list(data_topik, n=5)
                top_words = top_words_df['Kata'].tolist()
                kata_string = ', '.join(top_words)
                
                tabel_top_words.append({'Topik': topik, '5 Kata Populer': kata_string})
            
            if tabel_top_words:
                df_topik_kata = pd.DataFrame(tabel_top_words)
                
                # Tampilkan dalam Streamlit dengan styling
                st.markdown("### 📌 Tabel Topik & 5 Kata Populer")
                st.dataframe(
                    df_topik_kata.style.set_properties(**{
                        'text-align': 'left',
                        'white-space': 'pre-wrap'
                    }),
                    use_container_width=True
                )
            else:
                st.info("Tidak ada kata populer yang bisa ditampilkan.")



    
        else:
            st.subheader(f"🧩 Wordcloud Komentar: {selected_topik}")
            data_topik = df_filtered['cleaned'].dropna()
    
            if data_topik.empty:
                st.warning("Tidak ada komentar tersedia untuk topik ini.")
            else:
                get_wordcloud(data_topik, f"Wordcloud Komentar {selected_topik}")
    
                st.subheader(f"📌 Top Kata pada Topik: {selected_topik}")
                top_words = get_top_words(data_topik)
                plot_top_words(top_words, f"Top 10 Kata pada Topik {selected_topik}")


if __name__ == "__main__":
    main()
