"""
Module Giao Diện Tối Giản Sang Trọng (Minimalist Luxury - Matte Graphite Dark Mode)
Thiết kế chuyên biệt cho Hệ thống AI Chẩn đoán Bệnh Tôm (ConvNeXt-Tiny & U2-Net).
"""

import streamlit as st

MINIMALIST_LUXURY_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-graphite-deep: #0b0c0e;
        --bg-graphite-main: #111317;
        --bg-graphite-card: #171a20;
        --bg-graphite-elevated: #1e222a;
        --border-hairline: rgba(255, 255, 255, 0.07);
        --border-accent: rgba(212, 175, 55, 0.35);
        --accent-champagne: #d4af37;
        --accent-platinum: #e2e8f0;
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --status-healthy: #10b981;
        --status-warning: #f59e0b;
        --status-danger: #ef4444;
    }

    /* Tổng thể trang & Font chữ */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
    }

    /* Nền Đen Nhám (Matte Graphite Dark Mode) */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1d24 0%, #111317 55%, #0b0c0e 100%) !important;
        background-attachment: fixed !important;
    }

    /* Ẩn hoàn toàn các menu mặc định của Streamlit */
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    .stDeployButton,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        visibility: hidden !important;
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
        height: 2.5rem !important;
    }

    /* Không gian thoáng đãng chuẩn Minimalist Luxury */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    /* ==========================================================================
       TIÊU ĐỀ MINIMALIST LUXURY: THANH MẢNH, VIẾT HOA, KHOẢNG CÁCH CHỮ RỘNG
       ========================================================================== */
    .luxury-header-wrap {
        text-align: center;
        padding: 28px 20px 24px 20px;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--border-hairline);
    }

    .luxury-kicker {
        font-size: 10.5px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.38em;
        color: var(--accent-champagne);
        margin-bottom: 12px;
        display: block;
    }

    .luxury-main-title {
        font-size: 28px !important;
        font-weight: 200 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.26em !important;
        color: #ffffff !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }

    .luxury-subtitle {
        font-size: 12.5px;
        font-weight: 300;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        margin-top: 10px;
    }

    /* ==========================================================================
       SIDEBAR GRAPHITE TỐI GIẢN
       ========================================================================== */
    section[data-testid="stSidebar"] {
        background-color: #0d0f12 !important;
        border-right: 1px solid var(--border-hairline) !important;
    }

    .sidebar-minimal-header {
        font-size: 11px;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.25em;
        color: var(--accent-champagne);
        padding-bottom: 12px;
        margin-bottom: 18px;
        border-bottom: 1px solid var(--border-hairline);
    }

    .sidebar-group-label {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--text-muted);
        margin: 22px 0 10px 0;
    }

    /* ==========================================================================
       BỐ CỤC 3 TABS SANG TRỌNG ([KẾT QUẢ CHẨN ĐOÁN] - [PHÁC ĐỒ ĐIỀU TRỊ] - [THÔNG SỐ AI])
       ========================================================================== */
    div[data-baseweb="tab-list"] {
        background: #14171c !important;
        border: 1px solid var(--border-hairline) !important;
        border-radius: 10px !important;
        padding: 6px !important;
        gap: 6px !important;
        margin-top: 12px !important;
        margin-bottom: 28px !important;
        display: flex !important;
        justify-content: center !important;
    }

    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    button[data-baseweb="tab"] {
        flex: 1 1 0px !important;
        height: 46px !important;
        border-radius: 7px !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        transition: all 0.25s ease !important;
        margin: 0 !important;
    }

    button[data-baseweb="tab"] p {
        font-size: 12px !important;
        font-weight: 400 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.2em !important;
        margin: 0 !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #1e222b !important;
        border: 1px solid var(--border-accent) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: var(--accent-champagne) !important;
        font-weight: 500 !important;
    }

    /* ==========================================================================
       CARDS & CONTAINERS (GRAPHITE MATTE SURFACE)
       ========================================================================== */
    .graphite-card {
        background: var(--bg-graphite-card);
        border: 1px solid var(--border-hairline);
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .panel-label {
        font-size: 10.5px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--text-secondary);
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-hairline);
        padding-bottom: 8px;
    }

    .panel-step-index {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-champagne);
        font-size: 11px;
    }

    /* Kết quả chẩn đoán chính */
    .verdict-banner {
        background: linear-gradient(145deg, #191c23 0%, #13151a 100%);
        border: 1px solid var(--border-hairline);
        border-radius: 12px;
        padding: 26px 28px;
        margin-bottom: 20px;
    }

    .verdict-kicker {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.24em;
        color: var(--text-muted);
        margin-bottom: 6px;
    }

    .verdict-title {
        font-size: 24px;
        font-weight: 300;
        letter-spacing: 0.05em;
        color: #ffffff;
        margin: 0 0 10px 0;
    }

    .verdict-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: var(--accent-champagne);
    }

    /* Khung ảnh tinh tế */
    div[data-testid="stImage"] img {
        border-radius: 8px !important;
        border: 1px solid var(--border-hairline) !important;
    }

    /* Thanh chú giải nhiệt Grad-CAM tối giản */
    .minimal-heatmap-bar {
        height: 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, #1e3a8a 0%, #06b6d4 35%, #eab308 70%, #ef4444 100%);
        margin: 10px 0 6px 0;
    }

    .minimal-heatmap-labels {
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
    }

    /* Metric của Streamlit */
    div[data-testid="stMetric"] {
        background: var(--bg-graphite-card) !important;
        border: 1px solid var(--border-hairline) !important;
        border-radius: 10px !important;
        padding: 16px 18px !important;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 10.5px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.16em !important;
        color: var(--text-secondary) !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 22px !important;
        font-weight: 400 !important;
        color: #ffffff !important;
    }

    /* Nút bấm Tối Giản Sang Trọng */
    div.stButton > button[kind="primary"] {
        background: #1e222b !important;
        color: var(--accent-champagne) !important;
        font-weight: 400 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.24em !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.25s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: rgba(212, 175, 55, 0.12) !important;
        border-color: var(--accent-champagne) !important;
        color: #ffffff !important;
    }

    div.stDownloadButton > button {
        background: transparent !important;
        color: var(--text-primary) !important;
        font-size: 11.5px !important;
        font-weight: 400 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.18em !important;
        border: 1px solid var(--border-hairline) !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.2rem !important;
    }

    div.stDownloadButton > button:hover {
        border-color: var(--accent-champagne) !important;
        color: var(--accent-champagne) !important;
    }

    /* Khung tải ảnh tối giản */
    section[data-testid="stFileUploaderDropzone"] {
        background: #14171c !important;
        border: 1px dashed rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-champagne) !important;
        background: #181c22 !important;
    }
</style>
"""


def inject_luxury_ocean_theme(enable_animation: bool = False) -> dict:
    """Tiêm CSS Minimalist Luxury Graphite Dark Mode."""
    st.markdown(MINIMALIST_LUXURY_CSS, unsafe_allow_html=True)
    return {"gold_uri": "", "cyan_uri": ""}


def render_luxury_top_header(total_cases: int = 0, diseased_cases: int = 0, model_meta: dict = None):
    """Hiển thị Header Minimalist Luxury với tiêu đề thanh mảnh, viết hoa, khoảng cách chữ rộng."""
    st.markdown(
        """
        <div class="luxury-header-wrap">
            <span class="luxury-kicker">AQUA-XAI DIAGNOSTIC INTELLIGENCE</span>
            <h1 class="luxury-main-title">HỆ THỐNG CHẨN ĐOÁN BỆNH TÔM</h1>
            <div class="luxury-subtitle">CONVNEXT-TINY DEEP CLASSIFICATION &nbsp;•&nbsp; U2-NET SALIENT SEGMENTATION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
