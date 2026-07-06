import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Evaluasi Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_LABELS = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
}

CLASS_COLORS = {
    "cardboard": "#B7791F",
    "glass": "#0E9F6E",
    "metal": "#64748B",
    "paper": "#2563EB",
    "plastic": "#7C3AED",
    "trash": "#DC2626",
}

DATA = {
    "Kelas": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    "Precision": [0.89, 0.85, 0.81, 0.90, 0.80, 0.81],
    "Recall": [0.91, 0.90, 0.83, 0.89, 0.83, 0.50],
    "F1-Score": [0.90, 0.88, 0.82, 0.90, 0.81, 0.62],
    "Support": [69, 102, 88, 123, 89, 34],
}


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --ink: #102033;
                --muted: #5f6f83;
                --line: rgba(15, 23, 42, 0.10);
                --card: rgba(255, 255, 255, 0.92);
                --green: #0f766e;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(20, 184, 166, 0.15), transparent 30rem),
                    linear-gradient(135deg, #f8fbff 0%, #eef7f2 48%, #f7fbff 100%);
                color: var(--ink);
            }

            .block-container {
                max-width: 1180px;
                padding: 2rem 1.5rem 3rem;
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
                color: #ffffff;
                font-size: 1.35rem;
                line-height: 1.15;
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
                margin-bottom: 0.5rem;
                color: #ffffff;
            }

            .sidebar-mini {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.65rem;
                margin: 0.5rem 0;
                line-height: 1.35;
            }

            .hero,
            .metric-card,
            .panel,
            .class-card,
            .note-card {
                border: 1px solid var(--line);
                background: var(--card);
                box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08);
            }

            .hero {
                border-radius: 18px;
                padding: clamp(1.25rem, 3vw, 2.35rem);
                margin-bottom: 1.25rem;
                overflow: hidden;
                position: relative;
            }

            .hero::after {
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
                font-weight: 900;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.65rem;
            }

            .hero h1 {
                margin: 0;
                color: var(--ink);
                font-size: clamp(2rem, 5vw, 3.8rem);
                line-height: 1.04;
                letter-spacing: 0;
                max-width: 780px;
            }

            .hero p {
                margin: 1rem 0 0;
                color: var(--muted);
                font-size: clamp(1rem, 2vw, 1.13rem);
                line-height: 1.7;
                max-width: 760px;
                position: relative;
                z-index: 1;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.85rem;
                margin-bottom: 1.25rem;
            }

            .metric-card {
                border-radius: 16px;
                padding: 1rem;
            }

            .metric-card span {
                color: var(--muted);
                font-size: 0.83rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            .metric-card strong {
                display: block;
                margin-top: 0.45rem;
                color: var(--ink);
                font-size: 1.85rem;
                line-height: 1;
            }

            .panel {
                border-radius: 16px;
                padding: clamp(1rem, 2.2vw, 1.35rem);
                margin-bottom: 1.25rem;
            }

            .section-title {
                margin: 0 0 0.8rem;
                color: var(--ink);
                font-size: 1.35rem;
                font-weight: 900;
            }

            .helper-text {
                color: var(--muted);
                line-height: 1.65;
                margin: -0.2rem 0 1rem;
            }

            div[data-testid="stImage"] img {
                border-radius: 16px;
                border: 1px solid var(--line);
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
            }

            .class-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.85rem;
            }

            .class-card {
                border-radius: 16px;
                padding: 1rem;
                border-left: 6px solid var(--accent);
            }

            .class-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.8rem;
                margin-bottom: 0.85rem;
            }

            .class-name {
                color: var(--ink);
                font-weight: 900;
                font-size: 1.05rem;
            }

            .support {
                color: var(--muted);
                font-weight: 800;
                font-size: 0.86rem;
                white-space: nowrap;
            }

            .bar-row {
                display: grid;
                grid-template-columns: 82px minmax(80px, 1fr) 52px;
                gap: 0.6rem;
                align-items: center;
                margin: 0.55rem 0;
            }

            .bar-label,
            .bar-value {
                color: var(--muted);
                font-size: 0.86rem;
                font-weight: 800;
            }

            .bar-value {
                text-align: right;
                font-variant-numeric: tabular-nums;
            }

            .bar-track {
                height: 0.58rem;
                border-radius: 999px;
                overflow: hidden;
                background: rgba(15, 23, 42, 0.09);
            }

            .bar-fill {
                width: var(--value);
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 65%, #ffffff));
            }

            .note-card {
                border-radius: 16px;
                padding: 1rem;
                margin-top: 0.85rem;
                border-left: 6px solid #dc2626;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(254, 242, 242, 0.78));
            }

            .note-card strong {
                color: #991b1b;
                display: block;
                margin-bottom: 0.45rem;
            }

            .note-card p {
                color: #334155;
                line-height: 1.7;
                margin: 0;
            }

            @media screen and (max-width: 900px) {
                .block-container {
                    padding: 1rem 0.85rem 2rem;
                }

                .metric-grid,
                .class-grid {
                    grid-template-columns: 1fr 1fr;
                }
            }

            @media screen and (max-width: 640px) {
                .metric-grid,
                .class-grid {
                    grid-template-columns: 1fr;
                }

                .bar-row {
                    grid-template-columns: 78px minmax(70px, 1fr) 48px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df):
    weakest_class = df.sort_values("Recall").iloc[0]
    class_items = "".join(
        f"""
        <div class="sidebar-mini">
            <div>{CLASS_LABELS[row["Kelas"]]}</div>
            <strong>{row["F1-Score"] * 100:.0f}%</strong>
        </div>
        """
        for _, row in df.iterrows()
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <h2>Evaluasi Model</h2>
            <p>Ringkasan performa model klasifikasi sampah pada data validasi.</p>
        </div>

        <div class="sidebar-card">
            <strong>Highlight</strong>
            <div class="sidebar-mini"><div>Akurasi</div><strong>85%</strong></div>
            <div class="sidebar-mini"><div>Data validasi</div><strong>505</strong></div>
            <div class="sidebar-mini"><div>Jumlah kelas</div><strong>6</strong></div>
        </div>

        <div class="sidebar-card">
            <strong>F1-Score per Kelas</strong>
            {class_items}
        </div>

        <div class="sidebar-card">
            <strong>Perlu Perhatian</strong>
            <p>Kelas {CLASS_LABELS[weakest_class["Kelas"]]} memiliki recall {weakest_class["Recall"] * 100:.0f}%,
            sehingga penambahan data kelas ini akan paling membantu.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_cards(df):
    cards = []
    for _, row in df.iterrows():
        bars = "".join(
            f"""
            <div class="bar-row">
                <div class="bar-label">{name}</div>
                <div class="bar-track"><div class="bar-fill" style="--value: {row[name] * 100:.0f}%;"></div></div>
                <div class="bar-value">{row[name] * 100:.0f}%</div>
            </div>
            """
            for name in ["Precision", "Recall", "F1-Score"]
        )

        cards.append(
            f"""
            <div class="class-card" style="--accent: {CLASS_COLORS[row["Kelas"]]};">
                <div class="class-head">
                    <div class="class-name">{CLASS_LABELS[row["Kelas"]]}</div>
                    <div class="support">{row["Support"]} gambar</div>
                </div>
                {bars}
            </div>
            """
        )

    st.html(f'<div class="class-grid">{"".join(cards)}</div>')


df = pd.DataFrame(DATA)

inject_css()
render_sidebar(df)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Model Performance</div>
        <h1>Evaluasi model yang jelas, ringkas, dan mudah dibaca.</h1>
        <p>
            Halaman ini menampilkan performa model pada data validasi, mulai dari
            akurasi keseluruhan, distribusi metrik tiap kelas, sampai confusion matrix
            untuk melihat pola kesalahan prediksi.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metric-grid">
        <div class="metric-card"><span>Overall Accuracy</span><strong>85%</strong></div>
        <div class="metric-card"><span>Data Validasi</span><strong>505</strong></div>
        <div class="metric-card"><span>Total Dataset</span><strong>2.527</strong></div>
        <div class="metric-card"><span>Model</span><strong>MobileNetV2</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

matrix_col, note_col = st.columns([1.15, 0.85], gap="large")

with matrix_col:
    st.markdown('<h2 class="section-title">Confusion Matrix</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="helper-text">Matriks ini menunjukkan kelas asli dibandingkan dengan hasil prediksi model. Semakin kuat nilai pada diagonal utama, semakin baik performa klasifikasi.</p>',
        unsafe_allow_html=True,
    )
    st.image("confusion_matrix.png", width="stretch")

with note_col:
    st.markdown('<h2 class="section-title">Ringkasan Evaluasi</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="helper-text">
            Model sudah cukup kuat untuk kelas cardboard, glass, paper, plastic, dan metal.
            Area yang paling perlu diperbaiki adalah kelas trash karena jumlah data latihnya lebih sedikit.
        </p>
        <div class="note-card">
            <strong>Catatan penting</strong>
            <p>
                Kelas trash memiliki recall 50%. Artinya, sebagian gambar trash masih
                berpotensi terbaca sebagai kelas lain. Penambahan variasi data trash akan
                meningkatkan kestabilan model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<h2 class="section-title">Metrik per Kelas</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="helper-text">Precision, recall, dan F1-score ditampilkan dalam bentuk kartu agar perbandingan antar kelas lebih cepat dipahami.</p>',
    unsafe_allow_html=True,
)
render_class_cards(df)

st.markdown('<h2 class="section-title">Tabel Detail</h2>', unsafe_allow_html=True)
st.dataframe(
    df.assign(Kelas=df["Kelas"].map(CLASS_LABELS)),
    hide_index=True,
    width="stretch",
)
