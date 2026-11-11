"""
Cơ sở tri thức bệnh học tôm nuôi, Phác đồ điều trị chuyên sâu & Xuất Báo cáo HTML/PDF có nhúng ảnh Base64
Dùng cho Hệ thống AI Chẩn đoán Bệnh Tôm (U2-Net & ConvNeXt-Tiny)
"""

import io
import base64
from datetime import datetime
from PIL import Image

CLASS_DISPLAY_INFO = {
    "Healthy": {
        "vi_name": "Tôm Khỏe Mạnh (Healthy)",
        "short_name": "Khỏe mạnh",
        "badge_color": "#10b981",
        "bg_light": "#ecfdf5",
        "border_color": "#059669",
        "icon": "✅",
        "severity": "An toàn (Mức 0/3)",
        "severity_level": 0,
        "agent": "Không phát hiện tác nhân gây bệnh ngoại quan",
        "contagion_rate": "Không lây nhiễm",
        "summary": "Tôm có vỏ bóng mượt, màu sắc tự nhiên sắc nét, lá mang sạch sáng màu, không có đốm trắng dưới vỏ kitin hay vết hoại tử đen ở vùng mang.",
        "symptoms": [
            "Vỏ kitin sáng bóng, trong suốt tự nhiên, cứng cáp và không bị mềm vỏ.",
            "Vùng đầu ngực và các đốt bụng sạch, không xuất hiện đốm trắng tròn dưới vỏ.",
            "Hệ thống mang hai bên trong sáng, xếp đều, không bám bẩn hữu cơ hay chuyển màu đen/nâu.",
            "Phụ bộ (râu, chân bò, chân bơi, đuôi) nguyên vẹn, phản xạ bơi lội nhanh nhẹn."
        ],
        "causes": [
            "Môi trường nước ao nuôi ổn định, các chỉ số lý hóa nằm trong ngưỡng tối ưu.",
            "Chế độ dinh dưỡng cân đối, định kỳ bổ sung men vi sinh và khoáng chất."
        ],
        "immediate_actions": [
            "Duy trì chế độ cho ăn hiện tại, kiểm soát lượng thức ăn qua nhá (vó) để tránh dư thừa.",
            "Tiếp tục chạy quạt nước và sục khí đảm bảo Oxy hòa tan (DO) luôn ≥ 5.0 mg/L.",
            "Lấy mẫu kiểm tra định kỳ 3–5 ngày/lần thông qua tính năng 'Kiểm tra mẫu toàn ao'."
        ],
        "prevention": [
            "Bổ sung men tiêu hóa (Probiotics) + Vitamin C (2–3g/kg thức ăn) định kỳ 3 lần/tuần.",
            "Sử dụng chế phẩm vi sinh (Bacillus spp., Rhodobacter spp.) định kỳ 5–7 ngày/lần để làm sạch đáy ao.",
            "Duy trì độ kiềm (120–160 mg/L CaCO3) và pH ổn định (7.5–8.2) giúp tôm lột xác đồng loạt."
        ]
    },
    "WSSV": {
        "vi_name": "Bệnh Đốm Trắng (WSSV - White Spot Syndrome Virus)",
        "short_name": "Đốm trắng (WSSV)",
        "badge_color": "#ef4444",
        "bg_light": "#fef2f2",
        "border_color": "#dc2626",
        "icon": "🦠",
        "severity": "Báo động Đỏ - Cực kỳ nguy hiểm (Mức 3/3)",
        "severity_level": 3,
        "agent": "Virus WSSV (White Spot Syndrome Virus - giống Whispovirus, họ Nimaviridae)",
        "contagion_rate": "Cực nhanh (Tỷ lệ chết 80% - 100% chỉ sau 3 - 7 ngày bùng phát)",
        "summary": "Bệnh truyền nhiễm nguy hiểm bậc nhất trên tôm nuôi. Xuất hiện các đốm trắng tròn (0.5 - 2.0 mm) dưới lớp vỏ kitin vùng giáp đầu ngực và đốt bụng thứ 5, 6.",
        "symptoms": [
            "Xuất hiện nhiều đốm trắng tròn đường kính 0.5–2.0 mm dưới lớp vỏ kitin (rõ nhất ở giáp đầu ngực và đốt sống cuối).",
            "Tôm giảm ăn đột ngột, bơi lờ đờ dạt vào bờ hoặc nổi lên mặt nước vào sáng sớm và chiều tối.",
            "Thân tôm có thể chuyển sang màu hồng nhạt hoặc đỏ bầm do tế bào sắc tố giãn nở.",
            "Gan tụy nhợt nhạt, vỏ hơi mềm, tỷ lệ chết tăng đột biến chỉ sau 48–72 giờ phát hiện dấu hiệu đầu tiên."
        ],
        "causes": [
            "Lây nhiễm ngang qua nguồn nước cấp chưa xử lý triệt để, vật chủ trung gian (cua, còng, tôm tạp, chim nước).",
            "Lây truyền dọc từ tôm bố mẹ sang ấu trùng (Postlarvae) không qua xét nghiệm PCR.",
            "Bùng phát mạnh khi nhiệt độ nước biến động đột ngột (< 26°C), mưa lớn làm giảm độ mặn và pH gây sốc tôm."
        ],
        "immediate_actions": [
            "🚨 CÁCH LY NGAY LẬP TỨC ao nghi nhiễm: Tuyệt đối không xả nước trực tiếp ra kênh chung và dùng riêng toàn bộ dụng cụ (vợt, nhá, xô).",
            "Ngừng hoặc giảm 70–80% lượng thức ăn để hạn chế tôm khỏe ăn xác tôm bệnh (lây qua đường tiêu hóa cực nhanh).",
            "Vớt sạch tôm chết và tôm lờ đờ đem chôn hủy với vôi sống (CaO), tuyệt đối không vứt bừa bãi.",
            "Nếu tôm đã đạt kích cỡ thương phẩm: Khuyến nghị THU HOẠCH KHẨN CẤP để bảo toàn vốn vì bệnh do virus chưa có thuốc đặc trị.",
            "Nếu giữ ao: Nâng và duy trì nhiệt độ nước ổn định (nếu có mái che), tăng cường Oxy hòa tan > 6 mg/L, sát khuẩn nước bằng Iodine hoặc Potassium Monopersulfate (Virkon) liều an toàn."
        ],
        "prevention": [
            "100% con giống thả nuôi phải có giấy kiểm dịch âm tính với WSSV bằng phương pháp PCR.",
            "Lắp lưới chắn cua, còng, ếch nhái xung quanh bờ ao và lưới chống chim phía trên.",
            "Xử lý nước lắng kỹ bằng Chlorine (25–30 ppm) và chạy quạt sục khí 3–5 ngày trước khi cấp vào ao nuôi.",
            "Bổ sung Beta-glucan, Vitamin C liều cao trước các đợt không khí lạnh hoặc mưa bão."
        ]
    },
    "BG": {
        "vi_name": "Bệnh Đen Mang (BG - Black Gill Disease)",
        "short_name": "Đen mang (BG)",
        "badge_color": "#f59e0b",
        "bg_light": "#fffbeb",
        "border_color": "#d97706",
        "icon": "⚠️",
        "severity": "Cảnh báo Trung bình - Cao (Mức 2/3)",
        "severity_level": 2,
        "agent": "Nấm Fusarium spp., Vi khuẩn (Vibrio, Flavobacterium), Nguyên sinh động vật (Zoothamnium) & Khí độc đáy ao (NH3, NO2, H2S)",
        "contagion_rate": "Trung bình (Phụ thuộc chặt chẽ vào chất lượng đáy ao và môi trường nước)",
        "summary": "Lá mang tôm chuyển từ màu trắng ngà sang nâu xám hoặc đen sẫm do tích tụ melanin tại vùng mô mang bị viêm hoại tử, làm cản trở hô hấp và quá trình lột xác.",
        "symptoms": [
            "Tơ mang chuyển màu vàng nâu, nâu sẫm đến đen tuyền (dễ quan sát rõ qua lớp vỏ giáp hai bên đầu ngực).",
            "Mang bám nhiều chất hữu cơ, nhớt bẩn hoặc tảo tàn; các sợi mang bị ăn mòn, đứt gãy.",
            "Tôm bị suy hô hấp mãn tính: Thường nổi đầu hoặc dạt bờ vào sáng sớm khi Oxy hòa tan thấp.",
            "Tôm chậm lớn, khó lột xác hoặc chết rải rác sau khi lột xác do thiếu Oxy cục bộ."
        ],
        "causes": [
            "Đáy ao tích tụ nhiều bùn bã hữu cơ, thức ăn dư thừa, phân tôm và xác tảo tàn không được xi-phông sạch.",
            "Hàm lượng khí độc NH3 (> 0.1 mg/L), NO2 (> 5 mg/L), H2S tăng cao kết hợp kim loại nặng (Fe, Mn) kết tủa bám lên mang.",
            "Nhiễm nấm Fusarium solani, vi khuẩn cơ hội hoặc ký sinh trùng bám mang làm tổn thương tế bào biểu mô mang (kích hoạt phản ứng Melanin hóa làm đen mang)."
        ],
        "immediate_actions": [
            "Xi-phông (hút bùn) đáy ao ngay lập tức để loại bỏ chất thải hữu cơ và xác tảo.",
            "Thay 20% – 30% nước sạch đã qua xử lý từ ao lắng để làm loãng mật độ mầm bệnh và khí độc.",
            "Tăng cường tối đa thời gian chạy quạt nước và sục khí đáy; rải Oxy viên (Sodium Percarbonate) vào ban đêm.",
            "Diệt khuẩn và làm sạch mang bằng BKC (0.5–0.8 ppm) hoặc Glutaraldehyde / Iodine kết hợp Zeolite + Yucca hấp thụ khí độc.",
            "Sau 24h diệt khuẩn: Bổ sung khoáng chất (Ca, Mg, K) và Vitamin C để kích thích tôm đồng loạt lột xác, loại bỏ lớp vỏ mang cũ bị đen."
        ],
        "prevention": [
            "Quản lý chặt chẽ lượng thức ăn, xi-phông đáy ao hàng ngày đối với ao bạt công nghệ cao.",
            "Định kỳ sử dụng vi sinh phân hủy mùn bã hữu cơ và kiểm soát mật độ tảo (độ trong 30–40 cm).",
            "Kiểm tra khí độc NH3, NO2 mỗi ngày 1 lần và duy trì pH ở ngưỡng 7.6 – 8.0."
        ]
    },
    "BG_WSSV": {
        "vi_name": "Đồng Nhiễm Đen Mang & Đốm Trắng (BG + WSSV Co-infection)",
        "short_name": "Đồng nhiễm (BG + WSSV)",
        "badge_color": "#991b1b",
        "bg_light": "#fef2f2",
        "border_color": "#7f1d1d",
        "icon": "☠️",
        "severity": "Nguy kịch Tối đa (Mức 3+/3)",
        "severity_level": 4,
        "agent": "Đa tác nhân: Virus Đốm Trắng (WSSV) kết hợp Nấm/Vi khuẩn gây hoại tử Đen Mang trên nền đáy ao ô nhiễm",
        "contagion_rate": "Rất nhanh & Tỷ lệ hao hụt cực cao (Suy hô hấp + Nhiễm trùng huyết virus)",
        "summary": "Tình trạng bệnh kép cực kỳ nguy hiểm: Tôm vừa bị tổn thương mang nghiêm trọng (suy hô hấp, mất đề kháng) vừa bị virus WSSV tấn công toàn thân, dẫn đến chết hàng loạt nhanh hơn đơn nhiễm.",
        "symptoms": [
            "Xuất hiện đồng thời các đốm trắng dưới vỏ kitin (đầu ngực, thân) VÀ vùng mang chuyển màu đen sẫm/nâu đen.",
            "Tôm suy kiệt rất nhanh: Nổi đầu hàng loạt do vừa suy hô hấp ở mang vừa nhiễm virus toàn thân.",
            "Thân tôm đỏ bầm hoặc nhợt nhạt, vỏ mềm, râu đứt gãy, ruột trống rỗng không còn thức ăn.",
            "Tôm chết rải đáy và dạt bờ với tốc độ tăng theo cấp số nhân chỉ trong 24–48 giờ."
        ],
        "causes": [
            "Ao nuôi bị ô nhiễm đáy dẫn đến bệnh Đen mang trước, làm suy giảm hệ miễn dịch và tạo cửa ngõ cho virus WSSV bùng phát.",
            "Hoặc tôm đang ủ bệnh WSSV làm yếu sức, dẫn đến không tự làm sạch mang được, tạo điều kiện cho vi khuẩn/nấm và cặn bẩn tấn công vùng mang."
        ],
        "immediate_actions": [
            "🚨 BÁO ĐỘNG ĐỎ: Phong tỏa hoàn toàn ao bệnh, ngăn chặn lây lan sang các ao lân cận trong khu nuôi.",
            "TUYỆT ĐỐI KHÔNG kích thích lột xác hay thay nước ồ ạt vào lúc này (tôm đồng nhiễm bị suy hô hấp nặng sẽ chết đồng loạt ngay khi bị sốc môi trường).",
            "Tăng cường sục khí và rải Oxy viên tối đa để hỗ trợ hô hấp cho đàn tôm.",
            "Nếu tôm đạt cỡ thu hoạch: Tiến hành THU HOẠCH NGAY LẬP TỨC để giảm thiểu thiệt hại kinh tế.",
            "Nếu buộc phải hủy ao: Xử lý toàn bộ nước trong ao bằng Chlorine nồng độ cao (≥ 50 ppm) ngâm tối thiểu 7 ngày trước khi xả bỏ."
        ],
        "prevention": [
            "Không để bệnh Đen mang kéo dài: Khi phát hiện mang tôm hơi ngả vàng/nâu phải xử lý đáy ao ngay để không làm suy giảm miễn dịch.",
            "Tuân thủ nghiêm ngặt quy trình an toàn sinh học (Biosecurity) từ khâu chọn giống sạch bệnh PCR đến quản lý nguồn nước."
        ]
    }
}

WATER_QUALITY_STANDARDS = [
    {"Chỉ tiêu môi trường": "Nhiệt độ nước (°C)", "Ngưỡng tối ưu": "28.0 – 31.0 °C", "Ngưỡng nguy hiểm": "< 25°C (dễ bùng WSSV) hoặc > 33°C", "Tác động bệnh học": "Nhiệt độ thấp kích thích virus Đốm trắng (WSSV) nhân lên cực nhanh."},
    {"Chỉ tiêu môi trường": "Độ pH", "Ngưỡng tối ưu": "7.5 – 8.2 (dao động ngày < 0.5)", "Ngưỡng nguy hiểm": "< 7.0 hoặc > 8.8", "Tác động bệnh học": "pH cao làm tăng độc tính của khí NH3 gây tổn thương và đen mang."},
    {"Chỉ tiêu môi trường": "Oxy hòa tan - DO (mg/L)", "Ngưỡng tối ưu": "≥ 5.0 mg/L (Đáy ao ≥ 4.0)", "Ngưỡng nguy hiểm": "< 3.5 mg/L", "Tác động bệnh học": "Thiếu Oxy làm tôm Đen mang ngạt thở, tăng tỷ lệ chết khi nhiễm WSSV."},
    {"Chỉ tiêu môi trường": "Độ kiềm (mg/L CaCO3)", "Ngưỡng tối ưu": "120 – 160 mg/L", "Ngưỡng nguy hiểm": "< 80 mg/L", "Tác động bệnh học": "Kiềm thấp làm tôm mềm vỏ, khó lột xác để loại bỏ lớp mang đen."},
    {"Chỉ tiêu môi trường": "Khí độc NH3-N (mg/L)", "Ngưỡng tối ưu": "< 0.05 mg/L", "Ngưỡng nguy hiểm": "> 0.10 mg/L", "Tác động bệnh học": "Nguyên nhân hàng đầu gây ăn mòn biểu mô mang, dẫn tới bệnh Đen mang (BG)."},
    {"Chỉ tiêu môi trường": "Khí độc NO2-N (mg/L)", "Ngưỡng tối ưu": "< 1.0 mg/L (nước lợ)", "Ngưỡng nguy hiểm": "> 5.0 mg/L", "Tác động bệnh học": "Gắn kết với Hemocyanin trong máu tôm làm mất khả năng vận chuyển Oxy."},
    {"Chỉ tiêu môi trường": "Khí độc H2S (mg/L)", "Ngưỡng tối ưu": "0.00 mg/L (Không phát hiện)", "Ngưỡng nguy hiểm": "> 0.01 mg/L", "Tác động bệnh học": "Cực độc ở đáy ao bẩn, gây đen mang, thối mang và chết đột ngột."}
]


def pil_to_base64_uri(img_pil: Image.Image, max_width: int = 380) -> str:
    """Chuyển đổi ảnh PIL sang chuỗi Base64 Data URI để nhúng trực tiếp vào báo cáo HTML/PDF."""
    if img_pil is None:
        return ""
    img = img_pil.convert("RGB")
    w, h = img.size
    if w > max_width:
        new_h = max(1, int(round(h * (max_width / float(w)))))
        img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def generate_html_report(record: dict, images_dict: dict = None) -> str:
    """
    Tạo báo cáo ca bệnh định dạng HTML chuẩn in ấn (có thể Ctrl+P lưu thành PDF)
    kèm theo hình ảnh bằng chứng AI (Ảnh gốc ROI, Ảnh tách nền U2-Net, Bản đồ nhiệt Grad-CAM).
    """
    disease_code = record.get("pred_class", "Healthy")
    info = CLASS_DISPLAY_INFO.get(disease_code, CLASS_DISPLAY_INFO["Healthy"])
    probs = record.get("probabilities", {})
    timestamp = record.get("timestamp", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    filename = record.get("filename", "shrimp_sample.jpg")
    confidence = record.get("confidence", 0.0)
    entropy = record.get("entropy", 0.0)
    sharpness = record.get("sharpness", None)
    tta_used = "Bật (2x Multi-view)" if record.get("tta_used", True) else "Tắt (Single-view)"
    reliable_text = "Đạt chuẩn tin cậy" if record.get("is_reliable", True) else "Cảnh báo độ tin cậy thấp / Cần kiểm tra lại"

    images_html = ""
    if images_dict:
        b64_orig = pil_to_base64_uri(images_dict.get("orig_pil"))
        b64_clean = pil_to_base64_uri(images_dict.get("clean_pil"))
        b64_cam = pil_to_base64_uri(images_dict.get("gradcam_pil"))
        if b64_orig and b64_clean and b64_cam:
            images_html = f"""
            <h3>1. Bằng chứng Hình ảnh & Giải thích Mô hình AI (XAI - Grad-CAM)</h3>
            <div style="display:flex; gap:12px; justify-content:space-between; margin:14px 0 20px;">
                <div style="flex:1; text-align:center; border:1px solid #cbd5e1; border-radius:8px; padding:8px; background:#f8fafc;">
                    <img src="{b64_orig}" style="max-width:100%; border-radius:6px;" />
                    <div style="font-size:12px; font-weight:600; margin-top:6px;">1. Ảnh gốc & Khung ROI</div>
                </div>
                <div style="flex:1; text-align:center; border:1px solid #cbd5e1; border-radius:8px; padding:8px; background:#f8fafc;">
                    <img src="{b64_clean}" style="max-width:100%; border-radius:6px;" />
                    <div style="font-size:12px; font-weight:600; margin-top:6px;">2. Tách nền U2-Net & Auto-Crop</div>
                </div>
                <div style="flex:1; text-align:center; border:1px solid #cbd5e1; border-radius:8px; padding:8px; background:#f8fafc;">
                    <img src="{b64_cam}" style="max-width:100%; border-radius:6px;" />
                    <div style="font-size:12px; font-weight:600; margin-top:6px;">3. Bản đồ nhiệt Grad-CAM</div>
                </div>
            </div>
            """

    prob_rows = "".join([
        f"<tr><td style='padding:8px;border:1px solid #cbd5e1;'>{CLASS_DISPLAY_INFO.get(k, {}).get('vi_name', k)}</td>"
        f"<td style='padding:8px;border:1px solid #cbd5e1;font-weight:bold;'>{v:.2f}%</td></tr>"
        for k, v in probs.items()
    ])
    symptoms_html = "".join([f"<li>{s}</li>" for s in info["symptoms"]])
    actions_html = "".join([f"<li>{a}</li>" for a in info["immediate_actions"]])
    prevention_html = "".join([f"<li>{p}</li>" for p in info["prevention"]])
    sharp_str = f"{sharpness:.1f}" if sharpness is not None else "N/A"

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Phiếu Kết Quả Chẩn Đoán Bệnh Tôm - {filename}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b; max-width: 880px; margin: 20px auto; padding: 24px; border: 1px solid #cbd5e1; border-radius: 12px; }}
        .header {{ text-align: center; border-bottom: 2px solid #0284c7; padding-bottom: 14px; margin-bottom: 18px; }}
        .header h1 {{ margin: 0; color: #0f172a; font-size: 22px; }}
        .header p {{ margin: 6px 0 0; color: #475569; font-size: 13px; }}
        .badge {{ display: block; padding: 12px 18px; border-radius: 8px; background: {info['bg_light']}; border-left: 6px solid {info['border_color']}; color: #0f172a; font-weight: 700; font-size: 17px; margin: 14px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0 18px; font-size: 13.5px; }}
        th {{ background: #f1f5f9; text-align: left; padding: 8px; border: 1px solid #cbd5e1; }}
        td {{ padding: 8px; border: 1px solid #cbd5e1; }}
        h3 {{ color: #0369a1; border-left: 4px solid #0284c7; padding-left: 10px; margin-top: 18px; font-size: 16px; }}
        ul {{ line-height: 1.55; font-size: 13.5px; }}
        .footer {{ margin-top: 26px; font-size: 12px; color: #64748b; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
        @media print {{ body {{ border: none; margin: 0; padding: 10px; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🦐 PHIẾU KẾT QUẢ CHẨN ĐOÁN BỆNH TÔM BẰNG AI</h1>
        <p>Hệ thống Chẩn đoán Thông minh kết hợp Tách nền U2-Net, Auto-Crop ROI, Phân loại ConvNeXt-Tiny (TTA) & Giải thích Grad-CAM</p>
    </div>

    <table>
        <tr>
            <th>Mã mẫu / Tên file</th><td>{filename}</td>
            <th>Thời gian phân tích</th><td>{timestamp}</td>
        </tr>
        <tr>
            <th>Độ tin cậy AI (Confidence)</th><td><strong>{confidence:.2f}%</strong></td>
            <th>Độ bất định (Entropy) / Trạng thái</th><td>{entropy:.1f}% ({reliable_text})</td>
        </tr>
        <tr>
            <th>Độ sắc nét ảnh (Laplacian)</th><td>{sharp_str}</td>
            <th>Chế độ suy luận đa góc nhìn (TTA)</th><td>{tta_used}</td>
        </tr>
    </table>

    <div class="badge">
        {info['icon']} KẾT LUẬN: {info['vi_name']} — Mức độ: {info['severity']}
    </div>
    <p style="font-size:13.5px;"><strong>Tác nhân chính:</strong> {info['agent']}</p>
    <p style="font-size:13.5px;"><strong>Nhận định tổng quan:</strong> {info['summary']}</p>

    {images_html}

    <h3>2. Chi tiết xác suất dự đoán trên 4 lớp</h3>
    <table>
        <tr><th>Lớp chẩn đoán</th><th>Xác suất dự đoán (%)</th></tr>
        {prob_rows}
    </table>

    <h3>3. Triệu chứng bệnh học đặc trưng</h3>
    <ul>{symptoms_html}</ul>

    <h3>4. Phác đồ xử lý khẩn cấp tại ao nuôi</h3>
    <ul>{actions_html}</ul>

    <h3>5. Biện pháp phòng ngừa & An toàn sinh học</h3>
    <ul>{prevention_html}</ul>

    <div class="footer">
        Báo cáo được trích xuất tự động từ Hệ thống AI Chẩn đoán Bệnh Tôm (Đồ án 2 - ConvNeXt-Tiny & U2-Net).<br>
        Mẹo: Nhấn <strong>Ctrl + P</strong> trên trình duyệt để in hoặc lưu trực tiếp thành file PDF.
    </div>
</body>
</html>"""
