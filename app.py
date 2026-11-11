import os
import io
import json
import hashlib
from datetime import datetime

import streamlit as st
import torch
import numpy as np
import pandas as pd
import timm
from PIL import Image
from torchvision import transforms

from knowledge_base import (
    CLASS_DISPLAY_INFO,
    generate_html_report,
)
from xai_utils import (
    create_u2net_session,
    segment_shrimp_u2net,
    infer_and_gradcam,
    blend_gradcam_heatmap,
)

# ==============================================================================
# 1. CẤU HÌNH TRANG & CSS NHÚNG HIGH-END MEDICAL SCI-FI DASHBOARD (PALANTIR HUD)
# ==============================================================================
st.set_page_config(
    page_title="AQUA-XAI | Palantir Medical Sci-Fi Diagnostic HUD",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_medical_scifi_css():
    """
    Hệ thống CSS High-End Medical Sci-Fi Dashboard (Palantir Foundry / Biometric HUD):
    1. Glassmorphism: Kính mờ đa lớp (backdrop-filter: blur(18px)) + Viền phát sáng 1px.
    2. Keyframe Animations:
       - Nhịp tim Đỏ/Cam (pulsing glow) cho nút 'KÍCH HOẠT PHÂN TÍCH AI'.
       - Hover 3 khung ảnh: Phóng to mượt 1.02x (transform: scale(1.02)) + Viền phát sáng.
    3. Digital Telemetry Typography: Font JetBrains Mono & Orbitron kèm Text-Glow tương phản
       (Xanh Neon #00ff9d cho Khỏe mạnh, Đỏ/Cam #ff3b5c / #ff6b35 cho Bệnh).
    4. Pill Tabs: Thiết kế dạng viên thuốc bo tròn (999px) với chuyển màu mượt mà.
    """
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap');

            :root {
                --palantir-abyss: #06090f;
                --palantir-bg: #0a0f18;
                --glass-surface: rgba(14, 21, 35, 0.58);
                --glass-elevated: rgba(20, 30, 48, 0.72);
                --glass-border: rgba(56, 189, 248, 0.24);
                --glass-border-bright: rgba(56, 189, 248, 0.75);
                --neon-cyan: #38bdf8;
                --neon-green: #00ff9d;
                --neon-crimson: #ff3b5c;
                --neon-amber: #ff7b00;
                --text-primary: #f1f5f9;
                --text-secondary: #cbd5e1;
                --text-muted: #94a3b8;
            }

            /* Tổng thể Font chữ & Cỡ chữ lớn rõ nét */
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                color: var(--text-primary);
                font-size: 16px !important;
                -webkit-font-smoothing: antialiased;
            }

            p, li, label, span {
                font-size: 15.5px;
            }

            /* Nền Dark Mode Đen Nhám kết hợp Lưới Biometric Sci-Fi tinh tế */
            .stApp {
                background-color: var(--palantir-abyss) !important;
                background-image:
                    radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.10) 0%, transparent 50%),
                    radial-gradient(circle at 85% 20%, rgba(255, 59, 92, 0.06) 0%, transparent 40%),
                    linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px),
                    linear-gradient(180deg, #0b101b 0%, #070a12 60%, #05070c 100%) !important;
                background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px, 100% 100% !important;
                background-attachment: fixed !important;
            }

            /* Ẩn hoàn toàn Menu mặc định của Streamlit */
            #MainMenu,
            footer,
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
                height: 2rem !important;
            }

            .block-container {
                padding-top: 1.4rem !important;
                padding-bottom: 3.5rem !important;
                max-width: 1320px !important;
            }

            /* ==================================================================
               1. TIÊU ĐỀ SCI-FI LUXURY: THANH MẢNH, VIẾT HOA, KHOẢNG CÁCH RỘNG
               ================================================================== */
            .scifi-hero {
                position: relative;
                text-align: center;
                padding: 28px 24px 26px 24px;
                margin-bottom: 24px;
                background: rgba(12, 18, 30, 0.52);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                border: 1px solid rgba(56, 189, 248, 0.22);
                border-radius: 16px;
                box-shadow:
                    0 12px 40px rgba(0, 0, 0, 0.65),
                    inset 0 0 20px rgba(56, 189, 248, 0.05);
            }

            .scifi-kicker {
                font-family: 'JetBrains Mono', monospace;
                font-size: 12.5px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.42em;
                color: var(--neon-cyan);
                text-shadow: 0 0 12px rgba(56, 189, 248, 0.55);
                margin-bottom: 10px;
                display: block;
            }

            .scifi-title {
                font-size: 36px !important;
                font-weight: 200 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.28em !important;
                color: #ffffff !important;
                margin: 0 !important;
                line-height: 1.35 !important;
                text-shadow: 0 0 24px rgba(255, 255, 255, 0.18);
            }

            .scifi-subtitle {
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                font-weight: 400;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                color: var(--text-muted);
                margin-top: 12px;
            }

            /* ==================================================================
               2. HIỆU ỨNG NHỊP TIM (PULSING GLOW ĐỎ/CAM) CHO NÚT PHÂN TÍCH AI
               ================================================================== */
            @keyframes scifi-heartbeat-glow {
                0% {
                    box-shadow:
                        0 0 12px rgba(255, 59, 92, 0.45),
                        0 0 24px rgba(255, 123, 0, 0.25),
                        inset 0 0 12px rgba(255, 94, 54, 0.25);
                    border-color: rgba(255, 94, 54, 0.75);
                    transform: scale(1);
                }
                50% {
                    box-shadow:
                        0 0 22px rgba(255, 59, 92, 0.85),
                        0 0 42px rgba(255, 123, 0, 0.55),
                        inset 0 0 20px rgba(255, 94, 54, 0.45);
                    border-color: #ff7b00;
                    transform: scale(1.008);
                }
                100% {
                    box-shadow:
                        0 0 12px rgba(255, 59, 92, 0.45),
                        0 0 24px rgba(255, 123, 0, 0.25),
                        inset 0 0 12px rgba(255, 94, 54, 0.25);
                    border-color: rgba(255, 94, 54, 0.75);
                    transform: scale(1);
                }
            }

            div.stButton > button[kind="primary"],
            div.stButton > button {
                background: linear-gradient(135deg, rgba(255, 59, 92, 0.22) 0%, rgba(255, 123, 0, 0.26) 50%, rgba(255, 59, 92, 0.22) 100%) !important;
                backdrop-filter: blur(14px) !important;
                -webkit-backdrop-filter: blur(14px) !important;
                color: #ffffff !important;
                font-family: 'Orbitron', 'Inter', sans-serif !important;
                font-weight: 600 !important;
                font-size: 15px !important;
                text-transform: uppercase !important;
                letter-spacing: 0.22em !important;
                border: 1px solid rgba(255, 94, 54, 0.8) !important;
                border-radius: 999px !important;
                padding: 0.9rem 1.8rem !important;
                text-shadow: 0 0 12px rgba(255, 123, 0, 0.75) !important;
                animation: scifi-heartbeat-glow 2.4s infinite ease-in-out !important;
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
            }

            div.stButton > button[kind="primary"]:hover,
            div.stButton > button:hover {
                background: linear-gradient(135deg, rgba(255, 59, 92, 0.42) 0%, rgba(255, 123, 0, 0.48) 100%) !important;
                color: #ffffff !important;
                border-color: #ffffff !important;
                box-shadow:
                    0 0 30px rgba(255, 59, 92, 0.95),
                    0 0 55px rgba(255, 123, 0, 0.75) !important;
                transform: translateY(-2px) scale(1.012) !important;
            }

            /* ==================================================================
               3. TINH CHỈNH LAYOUT TABS: DẠNG PILLS BO TRÒN & CHUYỂN MÀU MƯỢT MÀ
               ================================================================== */
            div[data-testid="stTabs"] > div[data-baseweb="tab-list"],
            div[data-baseweb="tab-list"] {
                background: rgba(11, 17, 29, 0.65) !important;
                backdrop-filter: blur(20px) !important;
                -webkit-backdrop-filter: blur(20px) !important;
                border: 1px solid rgba(56, 189, 248, 0.25) !important;
                border-radius: 999px !important;
                padding: 8px !important;
                gap: 12px !important;
                margin-top: 20px !important;
                margin-bottom: 32px !important;
                box-shadow:
                    0 12px 34px rgba(0, 0, 0, 0.65),
                    inset 0 0 15px rgba(56, 189, 248, 0.06) !important;
                display: flex !important;
            }

            /* Ẩn hoàn toàn thanh gạch chân mặc định */
            div[data-baseweb="tab-highlight"],
            div[data-baseweb="tab-border"] {
                display: none !important;
            }

            /* Từng Tab thiết kế dạng Pill bo tròn */
            div[data-testid="stTabs"] button[data-baseweb="tab"],
            button[data-baseweb="tab"] {
                flex: 1 1 0px !important;
                height: 52px !important;
                border-radius: 999px !important;
                background: rgba(255, 255, 255, 0.02) !important;
                border: 1px solid rgba(255, 255, 255, 0.06) !important;
                color: var(--text-secondary) !important;
                margin: 0 !important;
                transition:
                    background 0.4s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.4s ease,
                    box-shadow 0.4s ease,
                    transform 0.3s ease,
                    color 0.3s ease !important;
            }

            button[data-baseweb="tab"] p {
                font-family: 'Orbitron', 'Inter', sans-serif !important;
                font-size: 14.5px !important;
                font-weight: 500 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.18em !important;
                margin: 0 !important;
                transition: all 0.35s ease !important;
            }

            button[data-baseweb="tab"]:hover {
                color: #ffffff !important;
                background: rgba(56, 189, 248, 0.10) !important;
                border-color: rgba(56, 189, 248, 0.45) !important;
                box-shadow: 0 0 18px rgba(56, 189, 248, 0.22) !important;
                transform: translateY(-1px) !important;
            }

            /* Pill Tab đang được chọn (Active State) */
            button[data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(135deg, rgba(56, 189, 248, 0.28) 0%, rgba(14, 116, 144, 0.42) 100%) !important;
                border: 1px solid var(--neon-cyan) !important;
                color: #ffffff !important;
                box-shadow:
                    0 0 24px rgba(56, 189, 248, 0.45),
                    inset 0 0 14px rgba(56, 189, 248, 0.25) !important;
            }

            button[data-baseweb="tab"][aria-selected="true"] p {
                color: #ffffff !important;
                font-weight: 700 !important;
                text-shadow: 0 0 12px rgba(56, 189, 248, 0.85) !important;
            }

            /* ==================================================================
               4. PHONG CÁCH GLASSMORPHISM (KÍNH MỜ + VIỀN 1PX PHÁT SÁNG NHẸ)
               ================================================================== */
            .glass-panel {
                background: var(--glass-surface);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                border: 1px solid var(--glass-border);
                border-radius: 14px;
                padding: 24px 26px;
                margin-bottom: 18px;
                box-shadow:
                    0 12px 36px rgba(0, 0, 0, 0.55),
                    0 0 12px rgba(56, 189, 248, 0.08),
                    inset 0 1px 0 rgba(255, 255, 255, 0.10);
                transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
            }

            .glass-panel:hover {
                border-color: rgba(56, 189, 248, 0.48);
                box-shadow:
                    0 14px 40px rgba(0, 0, 0, 0.65),
                    0 0 20px rgba(56, 189, 248, 0.16),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15);
            }

            .hud-header-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(56, 189, 248, 0.20);
                padding-bottom: 10px;
                margin-bottom: 14px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                color: var(--text-secondary);
            }

            .hud-tag {
                font-family: 'Orbitron', monospace;
                color: var(--neon-cyan);
                font-size: 13px;
                font-weight: 700;
                text-shadow: 0 0 10px rgba(56, 189, 248, 0.6);
            }

            /* ==================================================================
               5. KHUNG CHỨA 3 BỨC ẢNH (GLASSMORPHISM + HOVER SCALE 1.02X & GLOW)
               ================================================================== */
            div[data-testid="stImage"] {
                background: rgba(12, 19, 33, 0.58) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                border: 1px solid rgba(56, 189, 248, 0.28) !important;
                border-radius: 14px !important;
                padding: 10px !important;
                box-shadow:
                    0 10px 30px rgba(0, 0, 0, 0.55),
                    0 0 12px rgba(56, 189, 248, 0.08),
                    inset 0 0 12px rgba(56, 189, 248, 0.05) !important;
                overflow: hidden !important;
                transition:
                    transform 0.38s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.38s ease,
                    box-shadow 0.38s ease !important;
            }

            div[data-testid="stImage"] img {
                border-radius: 10px !important;
                transition: transform 0.38s cubic-bezier(0.16, 1, 0.3, 1) !important;
            }

            /* Khi hover vào 3 bức ảnh: Phóng to nhẹ 1.02x và viền container phát sáng */
            div[data-testid="stImage"]:hover {
                transform: scale(1.02) !important;
                border-color: var(--neon-cyan) !important;
                box-shadow:
                    0 16px 42px rgba(0, 0, 0, 0.75),
                    0 0 26px rgba(56, 189, 248, 0.55),
                    inset 0 0 18px rgba(56, 189, 248, 0.18) !important;
            }

            div[data-testid="stImage"]:hover img {
                transform: scale(1.02) !important;
            }

            /* ==================================================================
               6. TYPOGRAPHY KỸ THUẬT SỐ & TEXT-GLOW TƯƠNG PHẢN CAO
               ================================================================== */
            .digital-verdict-hud {
                background: rgba(14, 22, 38, 0.64);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(56, 189, 248, 0.32);
                border-radius: 16px;
                padding: 26px 32px;
                margin-bottom: 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 20px;
                box-shadow:
                    0 14px 40px rgba(0, 0, 0, 0.65),
                    0 0 22px rgba(56, 189, 248, 0.12),
                    inset 0 1px 0 rgba(255, 255, 255, 0.12);
            }

            .digital-number-healthy {
                font-family: 'Orbitron', 'JetBrains Mono', monospace !important;
                font-size: 38px !important;
                font-weight: 700 !important;
                color: var(--neon-green) !important;
                text-shadow:
                    0 0 14px rgba(0, 255, 157, 0.75),
                    0 0 30px rgba(0, 255, 157, 0.38) !important;
                letter-spacing: 0.05em;
            }

            .digital-number-danger {
                font-family: 'Orbitron', 'JetBrains Mono', monospace !important;
                font-size: 38px !important;
                font-weight: 700 !important;
                color: var(--neon-crimson) !important;
                text-shadow:
                    0 0 14px rgba(255, 59, 92, 0.85),
                    0 0 32px rgba(255, 123, 0, 0.45) !important;
                letter-spacing: 0.05em;
            }

            .digital-number-warning {
                font-family: 'Orbitron', 'JetBrains Mono', monospace !important;
                font-size: 38px !important;
                font-weight: 700 !important;
                color: var(--neon-amber) !important;
                text-shadow:
                    0 0 14px rgba(255, 123, 0, 0.85),
                    0 0 30px rgba(255, 123, 0, 0.42) !important;
                letter-spacing: 0.05em;
            }

            /* Thẻ xác suất (Probability Glass Cards) ở dưới cùng */
            .prob-glass-card {
                background: rgba(14, 22, 36, 0.58);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(56, 189, 248, 0.25);
                border-radius: 14px;
                padding: 18px 20px;
                box-shadow:
                    0 8px 28px rgba(0, 0, 0, 0.5),
                    0 0 12px rgba(56, 189, 248, 0.08),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);
                transition: all 0.3s ease;
            }

            .prob-glass-card:hover {
                transform: translateY(-3px);
                border-color: rgba(56, 189, 248, 0.65);
                box-shadow:
                    0 12px 32px rgba(0, 0, 0, 0.65),
                    0 0 20px rgba(56, 189, 248, 0.25);
            }

            .prob-card-label {
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--text-secondary);
                margin-bottom: 8px;
            }

            .prob-card-val-healthy {
                font-family: 'Orbitron', 'JetBrains Mono', monospace;
                font-size: 26px;
                font-weight: 700;
                color: var(--neon-green);
                text-shadow: 0 0 12px rgba(0, 255, 157, 0.7);
            }

            .prob-card-val-disease {
                font-family: 'Orbitron', 'JetBrains Mono', monospace;
                font-size: 26px;
                font-weight: 700;
                color: #ff5e36;
                text-shadow: 0 0 12px rgba(255, 94, 54, 0.75);
            }

            .prob-card-val-muted {
                font-family: 'Orbitron', 'JetBrains Mono', monospace;
                font-size: 26px;
                font-weight: 600;
                color: #94a3b8;
                text-shadow: 0 0 8px rgba(148, 163, 184, 0.25);
            }

            .prob-bar-track {
                width: 100%;
                height: 6px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 999px;
                margin-top: 12px;
                overflow: hidden;
            }

            /* Streamlit Native Metrics (ở Tab Thông Số AI) chuẩn Glassmorphism */
            div[data-testid="stMetric"] {
                background: rgba(14, 22, 36, 0.58) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                border: 1px solid rgba(56, 189, 248, 0.26) !important;
                border-radius: 14px !important;
                padding: 18px 22px !important;
                box-shadow:
                    0 8px 28px rgba(0, 0, 0, 0.5),
                    0 0 14px rgba(56, 189, 248, 0.08),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
                transition: all 0.3s ease !important;
            }

            div[data-testid="stMetric"]:hover {
                border-color: var(--neon-cyan) !important;
                box-shadow:
                    0 12px 32px rgba(0, 0, 0, 0.65),
                    0 0 22px rgba(56, 189, 248, 0.28) !important;
                transform: translateY(-2px) !important;
            }

            div[data-testid="stMetricLabel"] p {
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 12.5px !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.14em !important;
                color: var(--text-secondary) !important;
            }

            div[data-testid="stMetricValue"] {
                font-family: 'Orbitron', 'JetBrains Mono', monospace !important;
                font-size: 26px !important;
                font-weight: 700 !important;
                color: var(--neon-cyan) !important;
                text-shadow: 0 0 12px rgba(56, 189, 248, 0.6) !important;
            }

            /* Thang nhiệt Grad-CAM Sci-Fi */
            .gradcam-bar-scifi {
                height: 6px;
                border-radius: 999px;
                background: linear-gradient(90deg, #1e3a8a 0%, #00f5d4 35%, #fee440 70%, #ff0055 100%);
                margin: 12px 0 8px 0;
                box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
            }

            .gradcam-labels-scifi {
                display: flex;
                justify-content: space-between;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11.5px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: var(--text-secondary);
            }

            /* Sidebar & Khung Upload Kính mờ */
            section[data-testid="stSidebar"] {
                background: rgba(8, 12, 21, 0.92) !important;
                backdrop-filter: blur(20px) !important;
                border-right: 1px solid rgba(56, 189, 248, 0.22) !important;
            }

            section[data-testid="stFileUploaderDropzone"] {
                background: rgba(14, 21, 35, 0.55) !important;
                backdrop-filter: blur(16px) !important;
                border: 1px dashed rgba(56, 189, 248, 0.38) !important;
                border-radius: 14px !important;
                padding: 20px !important;
                transition: all 0.3s ease !important;
            }

            section[data-testid="stFileUploaderDropzone"]:hover {
                border-color: var(--neon-cyan) !important;
                background: rgba(20, 32, 54, 0.68) !important;
                box-shadow: 0 0 22px rgba(56, 189, 248, 0.22) !important;
            }

            div.stDownloadButton > button {
                background: rgba(14, 22, 38, 0.72) !important;
                backdrop-filter: blur(14px) !important;
                color: var(--neon-cyan) !important;
                font-family: 'Orbitron', 'Inter', sans-serif !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.18em !important;
                border: 1px solid rgba(56, 189, 248, 0.45) !important;
                border-radius: 999px !important;
                padding: 0.85rem 1.5rem !important;
                box-shadow: 0 0 16px rgba(56, 189, 248, 0.2) !important;
                transition: all 0.3s ease !important;
            }

            div.stDownloadButton > button:hover {
                background: rgba(56, 189, 248, 0.2) !important;
                border-color: var(--neon-cyan) !important;
                color: #ffffff !important;
                box-shadow: 0 0 26px rgba(56, 189, 248, 0.55) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_medical_scifi_css()

# ==============================================================================
# 2. QUẢN LÝ LỊCH SỬ CHẨN ĐOÁN & BỘ NHỚ ĐỆM PHIÊN (SESSION STATE)
# ==============================================================================
HISTORY_FILE = "diagnosis_history.json"


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history_list: list):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


if "history" not in st.session_state:
    st.session_state.history = load_history()

if "single_cache" not in st.session_state:
    st.session_state.single_cache = None


def append_history_record(record: dict):
    st.session_state.history.insert(0, record)
    save_history(st.session_state.history)


# ==============================================================================
# 3. TẢI MÔ HÌNH AI CONVNEXT & CACHED U2-NET ONNX SESSION (GIỮ NGUYÊN 100% LOGIC)
# ==============================================================================
@st.cache_resource
def load_model():
    model_path = "best_shrimp_convnext.pth"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names", ["BG", "BG_WSSV", "Healthy", "WSSV"])

    model = timm.create_model(
        "convnext_tiny", pretrained=False, num_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    file_size_mb = (
        os.path.getsize(model_path) / (1024 * 1024)
        if os.path.exists(model_path)
        else 0.0
    )

    model_meta = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "file_size_mb": file_size_mb,
        "checkpoint_keys": list(checkpoint.keys()),
        "epoch": checkpoint.get("epoch", "Đã hội tụ tối ưu"),
        "best_acc": checkpoint.get("best_acc", checkpoint.get("val_acc", None)),
    }
    return model, class_names, model_meta


@st.cache_resource
def get_cached_u2net_session():
    """Giữ phiên ONNX của U2-Net cố định trên RAM, giúp giảm thời gian tách nền từ ~1770ms xuống ~570ms."""
    return create_u2net_session("u2net")


try:
    model, class_names, model_meta = load_model()
    u2net_session = get_cached_u2net_session()
except Exception as e:
    st.error(
        f"❌ Không tìm thấy hoặc không thể đọc file mô hình `best_shrimp_convnext.pth`. "
        f"Vui lòng kiểm tra lại thư mục dự án! Chi tiết lỗi: {e}"
    )
    st.stop()

# Bộ tiền xử lý ảnh chuẩn ImageNet cho ConvNeXt-Tiny
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ==============================================================================
# 4. CÁC HÀM HIỂN THỊ GIAO DIỆN MEDICAL SCI-FI DASHBOARD (PALANTIR HUD)
# ==============================================================================
def render_scifi_header():
    """Hiển thị tiêu đề Medical Sci-Fi HUD với nền kính mờ Glassmorphism."""
    st.markdown(
        """
        <div class="scifi-hero">
            <span class="scifi-kicker">◈ PALANTIR BIOMETRIC HUD // AQUA-XAI TELEMETRY ◈</span>
            <h1 class="scifi-title">HỆ THỐNG NHẬN DIỆN BỆNH TÔM</h1>
            <div class="scifi-subtitle">CONVNEXT-TINY NEURAL CORE &nbsp;//&nbsp; U2-NET SALIENT SEGMENTATION &nbsp;//&nbsp; GRAD-CAM XAI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls() -> dict:
    """Thanh cấu hình ẩn gọn bên trái."""
    with st.sidebar:
        st.markdown(
            '<div class="hud-header-bar"><span>CẤU HÌNH HỆ THỐNG AI</span><span class="hud-tag">SYS</span></div>',
            unsafe_allow_html=True,
        )

        enable_rembg = st.toggle(
            "Tách nền U2-Net (ONNX)",
            value=True,
            help="Loại bỏ phông nền nhiễu bằng phiên U2-Net lưu sẵn trên RAM.",
        )
        auto_crop_roi = st.toggle(
            "Tự động cắt sát vùng tôm (ROI)",
            value=True,
            disabled=not enable_rembg,
        )
        enable_tta = st.toggle(
            "Suy luận đa góc nhìn (TTA 2x)",
            value=True,
        )

        bg_choice = st.selectbox(
            "Màu nền sau khi tách",
            options=[
                "Đen chuẩn (0, 0, 0)",
                "Trắng (255, 255, 255)",
                "Xám trung tính (128, 128, 128)",
            ],
            index=0,
            disabled=not enable_rembg,
        )
        bg_color_map = {
            "Đen chuẩn (0, 0, 0)": (0, 0, 0),
            "Trắng (255, 255, 255)": (255, 255, 255),
            "Xám trung tính (128, 128, 128)": (128, 128, 128),
        }
        selected_bg_color = bg_color_map[bg_choice]

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="hud-header-bar"><span>KIỂM SOÁT OOD & NHIỆT</span><span class="hud-tag">XAI</span></div>',
            unsafe_allow_html=True,
        )
        conf_threshold = st.slider(
            "Ngưỡng tin cậy tối thiểu (%)",
            min_value=30,
            max_value=95,
            value=60,
            step=5,
        )
        entropy_threshold = st.slider(
            "Ngưỡng Entropy tối đa (%)",
            min_value=40,
            max_value=95,
            value=72,
            step=2,
        )
        heatmap_opacity = st.slider(
            "Độ đậm lớp nhiệt Grad-CAM",
            min_value=0.20,
            max_value=0.75,
            value=0.45,
            step=0.05,
        )
        focus_on_fg = st.checkbox(
            "Giới hạn nhiệt trên thân tôm",
            value=True,
        )

    return {
        "enable_rembg": enable_rembg,
        "auto_crop_roi": auto_crop_roi,
        "enable_tta": enable_tta,
        "selected_bg_color": selected_bg_color,
        "conf_threshold": conf_threshold,
        "entropy_threshold": entropy_threshold,
        "heatmap_opacity": heatmap_opacity,
        "focus_on_fg": focus_on_fg,
    }


def render_tab_diagnosis(cached: dict, cfg: dict, live_gradcam_pil: Image.Image):
    """
    TAB 1: [KẾT QUẢ CHẨN ĐOÁN]
    - Khối kết quả Glassmorphism kèm con số kỹ thuật số phát sáng (Text-Glow).
    - 3 khung ảnh có hiệu ứng hover phóng to 1.02x và viền phát sáng.
    - 4 thẻ xác suất Glassmorphism ở dưới cùng với font chữ số kỹ thuật số.
    """
    seg_res = cached["seg_res"]
    xai_res = cached["xai_res"]
    pred_class = xai_res["pred_class"]
    confidence = xai_res["confidence"]
    entropy = xai_res["entropy"]
    fg_ratio = seg_res["foreground_ratio"]
    info = CLASS_DISPLAY_INFO.get(pred_class, CLASS_DISPLAY_INFO["Healthy"])

    ood_reasons = []
    if cfg["enable_rembg"] and fg_ratio < 2.5:
        ood_reasons.append(f"Tỷ lệ vùng tôm quá nhỏ ({fg_ratio:.1f}%)")
    if confidence < cfg["conf_threshold"]:
        ood_reasons.append(
            f"Độ tin cậy ({confidence:.1f}%) dưới ngưỡng ({cfg['conf_threshold']}%)"
        )
    if entropy > cfg["entropy_threshold"]:
        ood_reasons.append(
            f"Độ bất định Entropy cao ({entropy:.1f}% > {cfg['entropy_threshold']}%)"
        )
    is_reliable = len(ood_reasons) == 0
    cached["record"]["is_reliable"] = is_reliable

    # Chọn class Typography kỹ thuật số tương phản theo trạng thái bệnh
    is_healthy = pred_class == "Healthy"
    if is_healthy:
        digital_cls = "digital-number-healthy"
        glow_border = "rgba(0, 255, 157, 0.55)"
        status_color = "#00ff9d"
    elif pred_class in ["WSSV", "BG_WSSV"]:
        digital_cls = "digital-number-danger"
        glow_border = "rgba(255, 59, 92, 0.65)"
        status_color = "#ff3b5c"
    else:
        digital_cls = "digital-number-warning"
        glow_border = "rgba(255, 123, 0, 0.65)"
        status_color = "#ff7b00"

    status_text = (
        "BIOMETRIC VERIFIED // ĐẠT CHUẨN TIN CẬY"
        if is_reliable
        else "ANOMALY DETECTED // CẦN KIỂM TRA LẠI MẪU"
    )

    st.markdown(
        f"""
        <div class="digital-verdict-hud" style="border-color: {glow_border}; box-shadow: 0 14px 40px rgba(0,0,0,0.65), 0 0 25px {glow_border};">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:13px; letter-spacing:0.2em; color:{status_color}; text-transform:uppercase; margin-bottom:8px;">
                    ◈ {status_text}
                </div>
                <div style="font-size:28px; font-weight:300; letter-spacing:0.06em; text-transform:uppercase; color:#ffffff;">
                    {info['icon']} &nbsp;{info['vi_name']}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:12.5px; letter-spacing:0.22em; color:#94a3b8; text-transform:uppercase; margin-bottom:4px;">
                    ĐỘ TIN CẬY AI (CONFIDENCE)
                </div>
                <div class="{digital_cls}">{confidence:.2f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_reliable:
        st.warning("⚠️ **Cảnh báo kiểm định quang học OOD:** " + " | ".join(ood_reasons))

    # 3 Khung Ảnh (Hover phóng to 1.02x & Viền phát sáng)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="hud-header-bar">
                <span>ẢNH GỐC & KHUNG ROI</span>
                <span class="hud-tag">CAM // 01</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(seg_res["bbox_preview_pil"], use_container_width=True)

    with c2:
        st.markdown(
            """
            <div class="hud-header-bar">
                <span>TÁCH NỀN U2-NET</span>
                <span class="hud-tag">SEG // 02</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(seg_res["clean_pil"], use_container_width=True)

    with c3:
        st.markdown(
            """
            <div class="hud-header-bar">
                <span>BẢN ĐỒ NHIỆT GRAD-CAM</span>
                <span class="hud-tag">XAI // 03</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(live_gradcam_pil, use_container_width=True)
        st.markdown(
            """
            <div class="gradcam-bar-scifi"></div>
            <div class="gradcam-labels-scifi">
                <span>ÍT CHÚ Ý</span>
                <span>TRUNG BÌNH</span>
                <span>TẬP TRUNG CAO</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4 Thẻ Xác Suất (Probability Metrics) bọc Glassmorphism & Digital Font
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hud-header-bar">
            <span>PHÂN PHỐI XÁC SUẤT LƯỢNG TỬ (4 LỚP BỆNH HỌC)</span>
            <span class="hud-tag">SOFTMAX TELEMETRY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob_cols = st.columns(len(class_names))
    for idx, cls_code in enumerate(class_names):
        cls_prob = xai_res["probabilities"][cls_code]
        cls_short = CLASS_DISPLAY_INFO.get(cls_code, {}).get("short_name", cls_code)
        is_top = cls_code == pred_class

        if cls_code == "Healthy":
            val_class = "prob-card-val-healthy" if is_top else "prob-card-val-muted"
            bar_color = "#00ff9d" if is_top else "#38bdf8"
            card_border = (
                "border-color: rgba(0, 255, 157, 0.65); box-shadow: 0 0 20px rgba(0, 255, 157, 0.22);"
                if is_top
                else ""
            )
        else:
            val_class = "prob-card-val-disease" if is_top else "prob-card-val-muted"
            bar_color = "#ff3b5c" if is_top else "#38bdf8"
            card_border = (
                "border-color: rgba(255, 59, 92, 0.65); box-shadow: 0 0 20px rgba(255, 59, 92, 0.25);"
                if is_top
                else ""
            )

        with prob_cols[idx]:
            st.markdown(
                f"""
                <div class="prob-glass-card" style="{card_border}">
                    <div class="prob-card-label">{cls_short} [{cls_code}]</div>
                    <div class="{val_class}">{cls_prob:.2f}%</div>
                    <div class="prob-bar-track">
                        <div style="width:{min(max(cls_prob, 1.5), 100.0):.1f}%; height:100%; background:{bar_color}; box-shadow:0 0 10px {bar_color}; border-radius:999px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_tab_protocol(cached: dict, live_gradcam_pil: Image.Image = None):
    """
    TAB 2: [PHÁC ĐỒ ĐIỀU TRỊ]
    Các khối phác đồ bọc kính mờ Glassmorphism với viền 1px phát sáng nhẹ.
    """
    if cached is not None:
        pred_class = cached["xai_res"]["pred_class"]
        record = cached["record"]
        seg_res = cached["seg_res"]
        images_dict = {
            "orig_pil": seg_res["bbox_preview_pil"],
            "clean_pil": seg_res["clean_pil"],
            "gradcam_pil": live_gradcam_pil,
        }
    else:
        selected_cls = st.selectbox(
            "Tra cứu phác đồ điều trị theo nhóm bệnh:",
            options=list(CLASS_DISPLAY_INFO.keys()),
            format_func=lambda k: f"{CLASS_DISPLAY_INFO[k]['icon']} {CLASS_DISPLAY_INFO[k]['vi_name']}",
        )
        pred_class = selected_cls
        record = {
            "pred_class": pred_class,
            "filename": "Tra_cuu_phac_do.jpg",
            "confidence": 100.0,
            "entropy": 0.0,
            "is_reliable": True,
            "probabilities": {
                k: (100.0 if k == pred_class else 0.0) for k in CLASS_DISPLAY_INFO
            },
        }
        images_dict = None

    info = CLASS_DISPLAY_INFO.get(pred_class, CLASS_DISPLAY_INFO["Healthy"])

    st.markdown(
        f"""
        <div class="glass-panel" style="border-left: 4px solid {info['badge_color']};">
            <div class="hud-header-bar">
                <span>HỒ SƠ BỆNH HỌC &nbsp;//&nbsp; {info['severity']}</span>
                <span class="hud-tag">{pred_class}</span>
            </div>
            <div style="font-size:24px; font-weight:300; color:#ffffff; margin-bottom:10px;">
                {info['icon']} &nbsp;{info['vi_name']}
            </div>
            <div style="font-size:16px; color:#cbd5e1; line-height:1.7; margin-bottom:14px;">
                {info['summary']}
            </div>
            <div style="font-size:15px; color:#38bdf8; line-height:1.6;">
                <b>TÁC NHÂN:</b> <span style="color:#f3f4f6;">{info['agent']}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>TỐC ĐỘ LÂY LAN:</b> <span style="color:#f3f4f6;">{info['contagion_rate']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sym, col_act, col_prev = st.columns(3)

    with col_sym:
        symptoms_html = "".join(
            [f"<li style='margin-bottom:10px;'>{s}</li>" for s in info["symptoms"]]
        )
        st.markdown(
            f"""
            <div class="glass-panel" style="height:100%;">
                <div class="hud-header-bar">
                    <span>DẤU HIỆU NHẬN BIẾT</span>
                    <span class="hud-tag">SYM // 01</span>
                </div>
                <ul style="font-size:15.5px; color:#e5e7eb; padding-left:20px; line-height:1.7; margin:0;">
                    {symptoms_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_act:
        actions_html = "".join(
            [
                f"<li style='margin-bottom:10px;'>{a}</li>"
                for a in info["immediate_actions"]
            ]
        )
        st.markdown(
            f"""
            <div class="glass-panel" style="height:100%;">
                <div class="hud-header-bar">
                    <span>PHÁC ĐỒ XỬ LÝ TẠI AO</span>
                    <span class="hud-tag">ACT // 02</span>
                </div>
                <ul style="font-size:15.5px; color:#e5e7eb; padding-left:20px; line-height:1.7; margin:0;">
                    {actions_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_prev:
        prev_html = "".join(
            [f"<li style='margin-bottom:10px;'>{p}</li>" for p in info["prevention"]]
        )
        st.markdown(
            f"""
            <div class="glass-panel" style="height:100%;">
                <div class="hud-header-bar">
                    <span>PHÒNG NGỪA SINH HỌC</span>
                    <span class="hud-tag">BIO // 03</span>
                </div>
                <ul style="font-size:15.5px; color:#e5e7eb; padding-left:20px; line-height:1.7; margin:0;">
                    {prev_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    html_report = generate_html_report(record, images_dict=images_dict)
    st.download_button(
        label="📥 XUẤT PHIẾU BÁO CÁO Y KHOA (HTML / IN PDF)",
        data=html_report.encode("utf-8"),
        file_name=f"Phieu_Chan_Doan_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        use_container_width=True,
    )


def render_tab_ai_specs(cached: dict):
    """
    TAB 3: [THÔNG SỐ AI]
    Hiển thị Telemetry kỹ thuật số và thông số mô hình trong các khung Glassmorphism.
    """
    if cached is not None:
        seg_res = cached["seg_res"]
        xai_res = cached["xai_res"]
        iqa = seg_res["iqa"]
        orig_w, orig_h = cached["orig_size"]
        crop_w, crop_h = seg_res["clean_pil"].size

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BẤT ĐỊNH SHANNON ENTROPY", f"{xai_res['entropy']:.2f}%")
        m2.metric("THỜI GIAN TÁCH NỀN U2-NET", f"{seg_res['seg_time_ms']:.0f} ms")
        m3.metric("SUY LUẬN CONVNEXT-TINY", f"{xai_res['infer_time_ms']:.0f} ms")
        m4.metric("TỶ LỆ VÙNG THÂN TÔM (ROI)", f"{seg_res['foreground_ratio']:.1f}%")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        col_iqa, col_arch = st.columns(2)

        with col_iqa:
            st.markdown(
                f"""
                <div class="glass-panel">
                    <div class="hud-header-bar">
                        <span>KIỂM ĐỊNH CHẤT LƯỢNG ẢNH (IQA)</span>
                        <span class="hud-tag">OPTICAL QC</span>
                    </div>
                    <div style="font-size:15.5px; color:#e5e7eb; line-height:2.0;">
                        • Kích thước ảnh gốc: <code>{orig_w} × {orig_h} px</code><br>
                        • Kích thước vùng ROI sau cắt: <code>{crop_w} × {crop_h} px</code><br>
                        • Độ sắc nét (Laplacian Variance): <code>{iqa['sharpness']}</code><br>
                        • Độ sáng trung bình: <code>{iqa['brightness']} / 255</code><br>
                        • Tỷ lệ phản quang vỏ (Specular Glare): <code>{iqa['glare_ratio']}%</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_arch:
            st.markdown(
                f"""
                <div class="glass-panel">
                    <div class="hud-header-bar">
                        <span>KIẾN TRÚC MÔ HÌNH HỌC SÂU</span>
                        <span class="hud-tag">DEEP ENGINE</span>
                    </div>
                    <div style="font-size:15.5px; color:#e5e7eb; line-height:2.0;">
                        • Mạng phân loại chính: <code>ConvNeXt-Tiny (ImageNet Norm)</code><br>
                        • Mạng tách nền: <code>U2-Net Salient Object Detection (ONNX)</code><br>
                        • Tổng số tham số: <code>{model_meta['total_params']:,} params</code><br>
                        • Dung lượng trọng số: <code>{model_meta['file_size_mb']:.2f} MB</code><br>
                        • Tầng trích xuất Grad-CAM: <code>stages[-1] (7 × 7 × 768)</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MẠNG PHÂN LOẠI", "ConvNeXt-Tiny")
        m2.metric("MẠNG TÁCH NỀN", "U2-Net ONNX")
        m3.metric("TỔNG THAM SỐ", f"{model_meta['total_params'] / 1e6:.2f}M")
        m4.metric("TRỌNG SỐ MÔ HÌNH", f"{model_meta['file_size_mb']:.1f} MB")

        st.markdown(
            """
            <div class="glass-panel">
                <div class="hud-header-bar">
                    <span>THÔNG SỐ PIPELINE THỊ GIÁC MÁY TÍNH</span>
                    <span class="hud-tag">SPECIFICATIONS</span>
                </div>
                <div style="font-size:15.5px; color:#e5e7eb; line-height:2.0;">
                    • <b>Giai đoạn 1 (Salient Segmentation):</b> Tách nền U2-Net qua phiên ONNX lưu đệm RAM kết hợp tự động cắt sát khung bao thân tôm (Auto-Crop ROI + 10% lề).<br>
                    • <b>Giai đoạn 2 (Deep Feature Extraction):</b> Chuẩn hóa tensor <code>224 × 224</code> và suy luận đa góc nhìn (Test-Time Augmentation 2x Horizontal Flip) trên mạng <code>ConvNeXt-Tiny</code>.<br>
                    • <b>Giai đoạn 3 (Explainable AI & OOD Guard):</b> Trích xuất bản đồ nhiệt Grad-CAM từ tầng đặc trưng cuối <code>7 × 7 × 768</code> và kiểm soát độ bất định bằng Shannon Entropy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.history:
        with st.expander(
            f"📋 NHẬT KÝ PHIÊN CHẨN ĐOÁN ({len(st.session_state.history)} BẢN GHI)"
        ):
            df_hist = pd.DataFrame(st.session_state.history)[
                [
                    "timestamp",
                    "filename",
                    "vi_name",
                    "confidence",
                    "entropy",
                    "total_time_ms",
                ]
            ].rename(
                columns={
                    "timestamp": "Thời gian",
                    "filename": "Tên file",
                    "vi_name": "Kết luận",
                    "confidence": "Độ tin cậy (%)",
                    "entropy": "Entropy (%)",
                    "total_time_ms": "Tổng thời gian (ms)",
                }
            )
            st.dataframe(df_hist, use_container_width=True, hide_index=True)


# ==============================================================================
# 5. LUỒNG TRẢI NGHIỆM NGƯỜI DÙNG CHÍNH (CHỈ PHÂN TÍCH KHI NHẤN NÚT)
# ==============================================================================
render_scifi_header()
cfg = render_sidebar_controls()

input_col1, input_col2 = st.columns([2.2, 1])
with input_col2:
    input_mode = st.radio(
        "Nguồn ảnh đầu vào",
        options=["Tải ảnh từ thiết bị", "Chụp ảnh từ Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

source_image = None
source_filename = "specimen.jpg"
raw_bytes = None
current_img_hash = None

if input_mode == "Tải ảnh từ thiết bị":
    uploaded_file = st.file_uploader(
        "Tải ảnh mẫu tôm cần phân tích (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        source_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        source_filename = uploaded_file.name
else:
    cam_photo = st.camera_input("Chụp ảnh mẫu tôm", label_visibility="collapsed")
    if cam_photo is not None:
        raw_bytes = cam_photo.getvalue()
        source_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        source_filename = f"camera_{datetime.now().strftime('%H%M%S')}.jpg"

if source_image is not None and raw_bytes is not None:
    w_orig, h_orig = source_image.size
    current_img_hash = hashlib.md5(raw_bytes).hexdigest()[:12]
    config_sig = (
        f"{current_img_hash}_{cfg['enable_rembg']}_{cfg['auto_crop_roi']}_"
        f"{cfg['selected_bg_color']}_{cfg['enable_tta']}"
    )

    btn_run = st.button(
        "⚡ KÍCH HOẠT PHÂN TÍCH AI & TRÍCH XUẤT GRAD-CAM",
        type="primary",
        use_container_width=True,
    )

    if btn_run:
        with st.spinner(
            "Đang thực thi tách nền U2-Net & suy luận mạng sâu ConvNeXt-Tiny..."
        ):
            seg_res = segment_shrimp_u2net(
                source_image,
                bg_color=cfg["selected_bg_color"],
                enable_rembg=cfg["enable_rembg"],
                rembg_session=u2net_session,
                auto_crop_roi=cfg["auto_crop_roi"],
                padding_ratio=0.10,
                max_side=800,
            )

            xai_res = infer_and_gradcam(
                model=model,
                clean_pil=seg_res["clean_pil"],
                transform=transform,
                class_names=class_names,
                alpha_mask=seg_res["alpha_mask"],
                heatmap_opacity=cfg["heatmap_opacity"],
                focus_on_foreground=cfg["focus_on_fg"],
                enable_tta=cfg["enable_tta"],
            )

        pred_class = xai_res["pred_class"]
        confidence = xai_res["confidence"]
        entropy = xai_res["entropy"]
        fg_ratio = seg_res["foreground_ratio"]
        iqa = seg_res["iqa"]
        info = CLASS_DISPLAY_INFO.get(pred_class, CLASS_DISPLAY_INFO["Healthy"])
        reliable_now = (
            (confidence >= cfg["conf_threshold"])
            and (entropy <= cfg["entropy_threshold"])
            and (not cfg["enable_rembg"] or fg_ratio >= 2.5)
        )

        record = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "filename": source_filename,
            "pred_class": pred_class,
            "vi_name": info["vi_name"],
            "confidence": round(confidence, 2),
            "entropy": round(entropy, 2),
            "foreground_ratio": round(fg_ratio, 2),
            "sharpness": iqa["sharpness"],
            "tta_used": cfg["enable_tta"],
            "is_reliable": reliable_now,
            "probabilities": {
                k: round(v, 2) for k, v in xai_res["probabilities"].items()
            },
            "total_time_ms": round(
                seg_res["seg_time_ms"] + xai_res["infer_time_ms"], 1
            ),
        }
        append_history_record(record)

        st.session_state.single_cache = {
            "img_hash": current_img_hash,
            "config_sig": config_sig,
            "filename": source_filename,
            "orig_size": (w_orig, h_orig),
            "seg_res": seg_res,
            "xai_res": xai_res,
            "record": record,
        }

# ==============================================================================
# 6. CHIA BỐ CỤC 3 TAB DẠNG PILLS ([KẾT QUẢ CHẨN ĐOÁN] - [PHÁC ĐỒ ĐIỀU TRỊ] - [THÔNG SỐ AI])
# ==============================================================================
tab_diagnosis, tab_protocol, tab_ai_specs = st.tabs(
    [
        "KẾT QUẢ CHẨN ĐOÁN",
        "PHÁC ĐỒ ĐIỀU TRỊ",
        "THÔNG SỐ AI",
    ]
)

cached_data = st.session_state.single_cache
active_result = (
    cached_data
    if (
        cached_data is not None
        and current_img_hash is not None
        and cached_data.get("img_hash") == current_img_hash
    )
    else None
)

live_gradcam = None
if active_result is not None:
    live_gradcam, _ = blend_gradcam_heatmap(
        clean_pil=active_result["seg_res"]["clean_pil"],
        cam_norm=active_result["xai_res"]["cam_norm"],
        alpha_mask=active_result["seg_res"]["alpha_mask"],
        heatmap_opacity=cfg["heatmap_opacity"],
        focus_on_foreground=cfg["focus_on_fg"],
    )

with tab_diagnosis:
    if active_result is not None:
        render_tab_diagnosis(active_result, cfg, live_gradcam)
    elif source_image is not None:
        st.markdown(
            """
            <div class="glass-panel" style="text-align:center; padding: 28px 24px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:13px; letter-spacing:0.24em; color:#38bdf8; text-transform:uppercase; margin-bottom:8px;">
                    ◈ SPECIMEN LOADED // ĐÃ NẠP ẢNH MẪU VÀO BUỒNG QUÉT
                </div>
                <div style="font-size:16.5px; font-weight:300; color:#e5e7eb;">
                    Nhấn nút <b>"⚡ KÍCH HOẠT PHÂN TÍCH AI & TRÍCH XUẤT GRAD-CAM"</b> đang phát sáng phía trên để bắt đầu.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prev_c1, prev_c2, prev_c3 = st.columns([1, 1.2, 1])
        with prev_c2:
            st.image(
                source_image,
                caption=f"Ảnh mẫu chờ phân tích: {source_filename} ({w_orig}×{h_orig} px)",
                use_container_width=True,
            )
    else:
        st.markdown(
            """
            <div class="glass-panel" style="text-align:center; padding: 48px 24px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:13px; letter-spacing:0.26em; color:#38bdf8; text-transform:uppercase; margin-bottom:12px;">
                    ◈ AWAITING BIOMETRIC INPUT // CHỜ TÍN HIỆU HÌNH ẢNH
                </div>
                <div style="font-size:17px; font-weight:300; color:#cbd5e1;">
                    Vui lòng tải lên hình ảnh mẫu tôm ở khung phía trên rồi nhấn nút kích hoạt phân tích.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_protocol:
    render_tab_protocol(active_result, live_gradcam)

with tab_ai_specs:
    render_tab_ai_specs(active_result)
