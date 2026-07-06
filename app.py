import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Klasifikasi Sampah Otomatis",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SARAN = {
    "cardboard": "Lipat atau ratakan kardus agar tidak memakan tempat. Pastikan kering dan tidak terkena minyak atau makanan, lalu serahkan ke bank sampah atau pengepul kardus.",
    "glass": "Cuci bersih dan pastikan tidak ada sisa cairan. Bungkus dengan kertas jika pecah agar tidak melukai. Serahkan ke tempat daur ulang kaca dan hindari membuangnya ke tempat sampah biasa.",
    "metal": "Bersihkan dari sisa makanan atau cairan. Kaleng aluminium dan besi bisa dijual ke pengepul logam atau diserahkan ke bank sampah.",
    "paper": "Pastikan kertas kering dan tidak berminyak. Kertas yang terkena minyak atau makanan sulit didaur ulang. Kumpulkan dan serahkan ke bank sampah atau pengepul kertas.",
    "plastic": "Cuci bersih dan keringkan. Cek kode daur ulang di bagian bawah kemasan, lalu serahkan ke bank sampah atau drop box plastik terdekat.",
    "trash": "Sampah ini termasuk residu dan tidak dapat didaur ulang. Buang ke tempat sampah residu, pastikan terbungkus rapat, dan jangan dicampur dengan sampah daur ulang.",
}

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_LABELS = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
}

CLASS_ICONS = {
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️",
}

CLASS_COLORS = {
    "cardboard": "#B7791F",
    "glass": "#0E9F6E",
    "metal": "#64748B",
    "paper": "#2563EB",
    "plastic": "#7C3AED",
    "trash": "#DC2626",
}


@tf.keras.utils.register_keras_serializable()
class PreprocessInput(tf.keras.layers.Layer):
    def call(self, x):
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --bg: #f5f7fb;
                --ink: #102033;
                --muted: #5f6f83;
                --line: rgba(15, 23, 42, 0.10);
                --card: rgba(255, 255, 255, 0.92);
                --green: #0f766e;
                --lime: #65a30d;
                --blue: #2563eb;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(20, 184, 166, 0.16), transparent 32rem),
                    linear-gradient(135deg, #f8fbff 0%, #eef7f2 48%, #f7fbff 100%);
                color: var(--ink);
            }

            .block-container {
                max-width: 1180px;
                padding: 2.1rem 1.5rem 3rem;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(15, 118, 110, 0.95), rgba(16, 32, 51, 0.98)),
                    radial-gradient(circle at top right, rgba(132, 204, 22, 0.30), transparent 16rem);
            }

            [data-testid="stSidebar"] * {
                color: #eefdf8;
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: rgba(238, 253, 248, 0.82);
            }

            .sidebar-brand {
                padding: 0.9rem 0.25rem 1rem;
            }

            .sidebar-brand h2 {
                margin: 0;
                font-size: 1.35rem;
                line-height: 1.15;
                color: #ffffff;
            }

            .sidebar-brand p {
                margin: 0.55rem 0 0;
                line-height: 1.55;
            }

            .sidebar-card {
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 16px;
                padding: 0.9rem;
                margin: 0.8rem 0;
                background: rgba(255, 255, 255, 0.10);
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.16);
            }

            .sidebar-card strong {
                display: block;
                margin-bottom: 0.45rem;
                color: #ffffff;
            }

            .sidebar-step,
            .sidebar-mini {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                margin: 0.55rem 0;
                color: rgba(238, 253, 248, 0.88);
                line-height: 1.35;
            }

            .sidebar-step span:first-child,
            .sidebar-mini span:first-child {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                flex: 0 0 1.75rem;
                width: 1.75rem;
                height: 1.75rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.16);
                font-weight: 900;
                color: #ffffff;
            }

            .hero {
                display: grid;
                grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
                gap: 1.25rem;
                align-items: stretch;
                margin-bottom: 1.25rem;
            }

            .hero-main,
            .hero-side,
            .panel {
                border: 1px solid var(--line);
                background: var(--card);
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
                backdrop-filter: blur(16px);
                border-radius: 18px;
            }

            .hero-main {
                padding: clamp(1.35rem, 3vw, 2.4rem);
                min-height: 260px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow: hidden;
                position: relative;
            }

            .hero-main::after {
                content: "";
                position: absolute;
                inset: auto -4rem -7rem auto;
                width: 19rem;
                height: 19rem;
                border-radius: 999px;
                background: linear-gradient(135deg, rgba(15, 118, 110, 0.22), rgba(37, 99, 235, 0.10));
            }

            .eyebrow {
                color: var(--green);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.65rem;
            }

            .hero h1 {
                color: var(--ink);
                font-size: clamp(2rem, 5vw, 4.1rem);
                line-height: 1.02;
                letter-spacing: 0;
                margin: 0;
                max-width: 760px;
            }

            .hero p {
                color: var(--muted);
                font-size: clamp(1rem, 2vw, 1.15rem);
                line-height: 1.7;
                margin: 1rem 0 0;
                max-width: 720px;
            }

            .hero-stats {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 1.75rem;
                position: relative;
                z-index: 1;
            }

            .stat {
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 0.85rem 0.95rem;
                background: rgba(248, 250, 252, 0.74);
            }

            .stat strong {
                display: block;
                font-size: 1.4rem;
                color: var(--ink);
                line-height: 1;
            }

            .stat span {
                color: var(--muted);
                font-size: 0.84rem;
            }

            .hero-side {
                padding: 1.2rem;
                display: grid;
                align-content: center;
                gap: 0.8rem;
            }

            .category-pill {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                padding: 0.78rem 0.9rem;
                border: 1px solid var(--line);
                border-radius: 999px;
                background: rgba(248, 250, 252, 0.78);
                color: var(--ink);
                font-weight: 700;
            }

            .category-pill span:first-child {
                font-size: 1.18rem;
            }

            .section-title {
                margin: 0.35rem 0 0.85rem;
                color: var(--ink);
                font-size: 1.35rem;
                font-weight: 800;
            }

            .helper-text {
                color: var(--muted);
                line-height: 1.65;
                margin: -0.25rem 0 1rem;
            }

            .panel {
                padding: clamp(1rem, 2.2vw, 1.35rem);
                min-height: 100%;
            }

            .result-card {
                border-left: 6px solid var(--accent);
                border-radius: 16px;
                padding: 1.25rem;
                background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.86));
                border-top: 1px solid var(--line);
                border-right: 1px solid var(--line);
                border-bottom: 1px solid var(--line);
                box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
            }

            .result-label {
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .result-name {
                color: var(--ink);
                font-size: clamp(1.8rem, 4vw, 3rem);
                font-weight: 900;
                line-height: 1.05;
                margin: 0.45rem 0;
            }

            .confidence {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                border-radius: 999px;
                padding: 0.55rem 0.85rem;
                background: color-mix(in srgb, var(--accent) 13%, white);
                color: var(--accent);
                font-weight: 800;
            }

            .advice {
                margin-top: 1rem;
                padding: 1rem;
                border-radius: 14px;
                background: #f8fafc;
                border: 1px solid var(--line);
                color: #334155;
                line-height: 1.7;
            }

            .prob-row {
                display: grid;
                grid-template-columns: 130px minmax(90px, 1fr) 64px;
                gap: 0.8rem;
                align-items: center;
                margin: 0.72rem 0;
            }

            .prob-name {
                color: var(--ink);
                font-weight: 800;
                white-space: nowrap;
            }

            .prob-track {
                height: 0.7rem;
                border-radius: 999px;
                overflow: hidden;
                background: rgba(15, 23, 42, 0.09);
            }

            .prob-fill {
                height: 100%;
                width: var(--value);
                border-radius: inherit;
                background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 70%, #ffffff));
            }

            .prob-value {
                color: var(--muted);
                font-variant-numeric: tabular-nums;
                text-align: right;
                font-weight: 700;
            }

            div[data-testid="stFileUploader"] {
                border: 2px dashed rgba(15, 118, 110, 0.68);
                border-radius: 20px;
                padding: 1rem;
                background:
                    linear-gradient(135deg, rgba(240, 253, 250, 0.96), rgba(239, 246, 255, 0.94));
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.80), 0 16px 34px rgba(15, 118, 110, 0.10);
            }

            div[data-testid="stFileUploader"] section {
                border: 0;
                min-height: 210px;
                padding: 1.25rem;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }

            div[data-testid="stFileUploader"] section > div {
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                gap: 0.55rem;
            }

            div[data-testid="stFileUploader"] small {
                color: #475569 !important;
            }

            div[data-testid="stFileUploader"] button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 178px;
                min-height: 2.75rem;
                border: 0 !important;
                border-radius: 999px !important;
                background: linear-gradient(135deg, #0f766e, #2563eb) !important;
                color: #ffffff !important;
                font-weight: 900 !important;
                box-shadow: 0 12px 26px rgba(37, 99, 235, 0.25);
            }

            div[data-testid="stFileUploader"] button:hover {
                filter: brightness(1.05);
                transform: translateY(-1px);
            }

            div[data-testid="stImage"] img {
                border-radius: 16px;
                border: 1px solid var(--line);
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
            }

            .stAlert {
                border-radius: 14px;
            }

            @media screen and (max-width: 900px) {
                .block-container {
                    padding: 1rem 0.85rem 2rem;
                }

                .hero {
                    grid-template-columns: 1fr;
                }

                .hero-main {
                    min-height: auto;
                }
            }

            @media screen and (max-width: 640px) {
                .hero-stats {
                    grid-template-columns: 1fr;
                }

                .category-pill {
                    border-radius: 14px;
                }

                .prob-row {
                    grid-template-columns: 1fr 52px;
                    gap: 0.45rem 0.7rem;
                }

                .prob-track {
                    grid-column: 1 / -1;
                    grid-row: 2;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    category_html = "".join(
        f"""
        <div class="category-pill">
            <span>{CLASS_ICONS[name]} {CLASS_LABELS[name]}</span>
            <span>{name}</span>
        </div>
        """
        for name in CLASS_NAMES
    )

    st.html(
        f"""
        <div class="hero">
            <section class="hero-main">
                <div>
                    <div class="eyebrow">Smart Waste Classifier</div>
                    <h1>Klasifikasi sampah lebih cepat, rapi, dan mudah dipahami.</h1>
                    <p>
                        Unggah foto sampah, lalu aplikasi akan membaca kategorinya,
                        menampilkan tingkat keyakinan model, dan memberi saran penanganan
                        yang sesuai.
                    </p>
                </div>
                <div class="hero-stats">
                    <div class="stat"><strong>6</strong><span>Kategori sampah</span></div>
                    <div class="stat"><strong>224</strong><span>Ukuran input model</span></div>
                    <div class="stat"><strong>AI</strong><span>MobileNetV2 model</span></div>
                </div>
            </section>
            <div class="hero-side">
                {category_html}
            </div>
        </div>
        """
    )


def render_sidebar():
    category_items = "".join(
        f"""
        <div class="sidebar-mini">
            <span>{CLASS_ICONS[name]}</span>
            <div>{CLASS_LABELS[name]}</div>
        </div>
        """
        for name in CLASS_NAMES
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <h2>♻️ Waste Classifier</h2>
            <p>Dashboard prediksi sampah berbasis AI dengan hasil yang cepat dan mudah dibaca.</p>
        </div>

        <div class="sidebar-card">
            <strong>Alur Penggunaan</strong>
            <div class="sidebar-step"><span>1</span><div>Upload foto sampah.</div></div>
            <div class="sidebar-step"><span>2</span><div>Tunggu model membaca gambar.</div></div>
            <div class="sidebar-step"><span>3</span><div>Lihat kategori dan saran penanganan.</div></div>
        </div>

        <div class="sidebar-card">
            <strong>Kategori Model</strong>
            {category_items}
        </div>

        <div class="sidebar-card">
            <strong>Tips Foto</strong>
            <p>Gunakan gambar terang, objek terlihat utuh, dan hindari terlalu banyak benda lain di latar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(predicted_class, confidence):
    label = CLASS_LABELS[predicted_class]
    icon = CLASS_ICONS[predicted_class]
    accent = CLASS_COLORS[predicted_class]

    st.html(
        f"""
        <div class="result-card" style="--accent: {accent};">
            <div class="result-label">Hasil prediksi</div>
            <div class="result-name">{icon} {label}</div>
            <div class="confidence">Confidence {confidence:.2f}%</div>
            <div class="advice">{SARAN[predicted_class]}</div>
        </div>
        """
    )


def render_probabilities(predictions):
    sorted_probs = sorted(
        zip(CLASS_NAMES, predictions[0]),
        key=lambda item: item[1],
        reverse=True,
    )

    rows = []
    for class_name, probability in sorted_probs:
        percent = float(probability) * 100
        rows.append(
            f"""
            <div class="prob-row" style="--accent: {CLASS_COLORS[class_name]};">
                <div class="prob-name">{CLASS_ICONS[class_name]} {CLASS_LABELS[class_name]}</div>
                <div class="prob-track"><div class="prob-fill" style="--value: {percent:.2f}%;"></div></div>
                <div class="prob-value">{percent:.2f}%</div>
            </div>
            """
        )

    st.html("".join(rows))


def preprocess_image(image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    return np.expand_dims(img_array, axis=0)


inject_css()
render_sidebar()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Model gagal dimuat. Pastikan file best_model.keras berada di folder aplikasi. Detail: {exc}")
    st.stop()

render_hero()

left_col, right_col = st.columns([0.95, 1.05], gap="large")

with left_col:
    st.markdown('<h2 class="section-title">Upload Gambar</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="helper-text">Gunakan foto dengan objek sampah terlihat jelas, pencahayaan cukup, dan latar tidak terlalu ramai.</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Pilih gambar sampah",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Belum ada gambar. Upload file JPG, JPEG, atau PNG untuk mulai klasifikasi.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", width="stretch")

with right_col:
    st.markdown('<h2 class="section-title">Analisis Model</h2>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown(
            '<p class="helper-text">Hasil prediksi, saran penanganan, dan distribusi probabilitas akan tampil di sini setelah gambar diupload.</p>',
            unsafe_allow_html=True,
        )
    else:
        img_array = preprocess_image(image)

        with st.spinner("Menganalisis gambar..."):
            predictions = model.predict(img_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        render_result(predicted_class, confidence)

        st.markdown('<h2 class="section-title" style="margin-top: 1.25rem;">Probabilitas per Kelas</h2>', unsafe_allow_html=True)
        render_probabilities(predictions)
