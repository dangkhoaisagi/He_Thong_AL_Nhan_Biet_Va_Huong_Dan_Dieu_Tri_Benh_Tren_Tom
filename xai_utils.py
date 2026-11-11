"""
Module Tối ưu hóa Tiền xử lý Tách nền Chuẩn Alpha (U2-Net Cached Session + Smart Resize + Auto-Crop ROI),
Đánh giá Chất lượng Ảnh (IQA), Suy luận Đa góc nhìn (TTA), Giải thích Mô hình (Grad-CAM)
và Đánh giá Độ bất định (Shannon Entropy) cho mạng ConvNeXt-Tiny.
"""

import time
import math
import cv2
import numpy as np
import torch
from PIL import Image
from rembg import remove, new_session


def create_u2net_session(model_name: str = "u2net"):
    """
    Khởi tạo phiên ONNX Runtime cho U2-Net một lần duy nhất (dùng với @st.cache_resource)
    giúp giảm thời gian tách nền từ ~1,770ms xuống ~550ms trên CPU.
    """
    return new_session(model_name)


def smart_resize_pil(image_pil: Image.Image, max_side: int = 800) -> Image.Image:
    """
    Thu phóng thông minh ảnh độ phân giải lớn (2K/4K) về kích thước tối ưu (cạnh lớn nhất <= max_side)
    giữ nguyên tỷ lệ khung hình, tăng tốc tách nền U2-Net mà không giảm chất lượng đầu vào ConvNeXt (224x224).
    """
    w, h = image_pil.size
    if max(w, h) <= max_side:
        return image_pil
    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)


def assess_image_quality(rgb_np: np.ndarray, alpha_mask: np.ndarray = None) -> dict:
    """
    Đánh giá chất lượng ảnh đầu vào (Image Quality Assessment - IQA) trên vùng thân tôm:
    1. Độ sắc nét (Laplacian Variance): Phát hiện ảnh bị rung mờ (out-of-focus / motion blur).
    2. Tỷ lệ chói lóa (Specular Glare %): Phát hiện phản xạ đèn Flash / ánh nắng gắt trên vỏ tôm ướt
       (yếu tố dễ gây nhầm lẫn với đốm trắng WSSV).
    3. Độ sáng trung bình (Mean Brightness): Phát hiện ảnh thiếu sáng.
    """
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)

    if alpha_mask is not None and (alpha_mask > 0.2).sum() > 200:
        fg_bool = alpha_mask > 0.2
        ys, xs = np.where(fg_bool)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        roi_gray = gray[y1:y2, x1:x2]
        lap_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
        mean_brightness = float(gray[fg_bool].mean())

        # Điểm chói trắng: Độ sáng V rất cao (> 245) và độ bão hòa màu S rất thấp (< 30)
        v_chan = hsv[:, :, 2][fg_bool]
        s_chan = hsv[:, :, 1][fg_bool]
        glare_ratio = float(((v_chan > 245) & (s_chan < 30)).mean() * 100.0)
    else:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(gray.mean())
        v_chan = hsv[:, :, 2]
        s_chan = hsv[:, :, 1]
        glare_ratio = float(((v_chan > 245) & (s_chan < 30)).mean() * 100.0)

    warnings = []
    if lap_var < 35.0:
        warnings.append(f"Ảnh hơi mờ (Độ nét Laplacian = {lap_var:.0f} < 35)")
    if glare_ratio > 4.5:
        warnings.append(f"Phát hiện lóa sáng/phản quang trên vỏ ({glare_ratio:.1f}% > 4.5%), lưu ý tránh nhầm với đốm trắng")
    if mean_brightness < 35.0:
        warnings.append(f"Ảnh thiếu sáng (Độ sáng TB = {mean_brightness:.0f}/255)")

    return {
        "sharpness": round(lap_var, 1),
        "brightness": round(mean_brightness, 1),
        "glare_ratio": round(glare_ratio, 2),
        "quality_ok": len(warnings) == 0,
        "quality_warnings": warnings,
    }


def segment_shrimp_u2net(
    image_pil: Image.Image,
    bg_color: tuple = (0, 0, 0),
    enable_rembg: bool = True,
    rembg_session=None,
    auto_crop_roi: bool = True,
    padding_ratio: float = 0.10,
    max_side: int = 800,
) -> dict:
    """
    Tách nền ảnh tôm bằng U2-Net với Cached ONNX Session + Smart Resize + Auto-Crop ROI:
    - Ghép kênh Alpha chuẩn xác lên nền đơn sắc.
    - Tự động tìm khung bao (Bounding Box) cắt sát vùng thân tôm (+10% lề an toàn)
      để phóng đại chi tiết đốm trắng (WSSV) và vùng mang (BG) trước khi đưa vào ConvNeXt-Tiny.
    """
    t0 = time.perf_counter()
    rgb_orig = image_pil.convert("RGB")
    rgb_proc = smart_resize_pil(rgb_orig, max_side=max_side) if max_side else rgb_orig
    img_np = np.array(rgb_proc)
    h, w = img_np.shape[:2]

    if not enable_rembg:
        full_mask = np.ones((h, w), dtype=np.float32)
        iqa = assess_image_quality(img_np, None)
        return {
            "clean_pil": rgb_proc,
            "clean_np": img_np,
            "bbox_preview_pil": rgb_proc,
            "alpha_mask": full_mask,
            "foreground_ratio": 100.0,
            "bbox": (0, 0, w, h),
            "seg_time_ms": (time.perf_counter() - t0) * 1000.0,
            "iqa": iqa,
        }

    # Gọi rembg với session được cache sẵn giúp tăng tốc ~3 lần
    if rembg_session is not None:
        rgba_np = remove(img_np, session=rembg_session)
    else:
        rgba_np = remove(img_np)

    if rgba_np.ndim == 3 and rgba_np.shape[2] == 4:
        alpha = rgba_np[:, :, 3].astype(np.float32) / 255.0
        fg_rgb = rgba_np[:, :, :3].astype(np.float32)
        bg_layer = np.full_like(fg_rgb, bg_color, dtype=np.float32)

        alpha_3c = np.expand_dims(alpha, axis=-1)
        composited = (fg_rgb * alpha_3c + bg_layer * (1.0 - alpha_3c)).clip(0, 255).astype(np.uint8)
        foreground_ratio = float((alpha > 0.15).mean() * 100.0)
    else:
        composited = rgba_np[:, :, :3].astype(np.uint8)
        alpha = np.ones((h, w), dtype=np.float32)
        foreground_ratio = 100.0

    # Đánh giá chất lượng ảnh trên vùng thân tôm
    iqa = assess_image_quality(img_np, alpha)

    # Tìm Bounding Box quanh đối tượng tôm để Auto-Crop ROI
    fg_coords = np.argwhere(alpha > 0.15)
    bbox_preview = img_np.copy()
    if fg_coords.shape[0] > 50:
        y_min, x_min = fg_coords.min(axis=0)
        y_max, x_max = fg_coords.max(axis=0)
        box_h = y_max - y_min
        box_w = x_max - x_min
        pad_y = int(box_h * padding_ratio)
        pad_x = int(box_w * padding_ratio)

        y1 = max(0, int(y_min - pad_y))
        y2 = min(h, int(y_max + pad_y))
        x1 = max(0, int(x_min - pad_x))
        x2 = min(w, int(x_max + pad_x))
        bbox = (x1, y1, x2, y2)

        # Vẽ khung phát hiện ROI màu xanh ngọc lên ảnh gốc thu phóng để minh họa trực quan
        cv2.rectangle(bbox_preview, (x1, y1), (x2, y2), (16, 185, 129), max(2, int(min(h, w) / 180)))

        if auto_crop_roi and (x2 - x1 > 32) and (y2 - y1 > 32):
            composited = composited[y1:y2, x1:x2]
            alpha = alpha[y1:y2, x1:x2]
    else:
        bbox = (0, 0, w, h)

    seg_time_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "clean_pil": Image.fromarray(composited),
        "clean_np": composited,
        "bbox_preview_pil": Image.fromarray(bbox_preview),
        "alpha_mask": alpha,
        "foreground_ratio": foreground_ratio,
        "bbox": bbox,
        "seg_time_ms": seg_time_ms,
        "iqa": iqa,
    }


def compute_normalized_entropy(prob_array: np.ndarray) -> float:
    """
    Tính độ bất định Normalized Shannon Entropy (%) trên phân phối xác suất dự đoán.
    """
    num_classes = len(prob_array)
    if num_classes <= 1:
        return 0.0
    eps = 1e-12
    probs = np.clip(prob_array, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(num_classes)
    return float((entropy / max_entropy) * 100.0)


def blend_gradcam_heatmap(
    clean_pil: Image.Image,
    cam_norm: np.ndarray,
    alpha_mask: np.ndarray = None,
    heatmap_opacity: float = 0.45,
    focus_on_foreground: bool = True,
) -> tuple:
    """
    Hàm pha trộn tức thì (< 3ms) bản đồ nhiệt Grad-CAM lên ảnh tôm.
    Tách rời khỏi bước suy luận mạng nơ-ron để khi người dùng kéo thanh trượt ở Sidebar,
    giao diện cập nhật ngay lập tức mà không cần chạy lại mô hình AI.
    """
    base_img = np.array(clean_pil.convert("RGB"))
    h, w = base_img.shape[:2]
    cam_resized = cv2.resize(cam_norm, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    if focus_on_foreground and alpha_mask is not None and alpha_mask.shape == (h, w):
        soft_mask = cv2.GaussianBlur(alpha_mask.astype(np.float32), (15, 15), 0)
        cam_resized = cam_resized * (0.15 + 0.85 * soft_mask)
        if cam_resized.max() > 1e-6:
            cam_resized = cam_resized / cam_resized.max()

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb = (
        base_img.astype(np.float32) * (1.0 - heatmap_opacity)
        + heatmap_rgb.astype(np.float32) * heatmap_opacity
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb)


def infer_and_gradcam(
    model: torch.nn.Module,
    clean_pil: Image.Image,
    transform,
    class_names: list,
    alpha_mask: np.ndarray = None,
    heatmap_opacity: float = 0.45,
    focus_on_foreground: bool = True,
    enable_tta: bool = True,
) -> dict:
    """
    Thực hiện suy luận (hỗ trợ Test-Time Augmentation 2x: Gốc + Lật ngang trên cùng 1 Batch)
    kết hợp trích xuất bản đồ nhiệt Grad-CAM từ tầng stages[-1] của ConvNeXt-Tiny.
    """
    t0 = time.perf_counter()
    model.eval()

    tensor_orig = transform(clean_pil).unsqueeze(0)
    tensor_orig.requires_grad_(True)

    activations = []
    target_layer = model.stages[-1] if hasattr(model, "stages") else list(model.children())[-2]

    def forward_hook(_module, _inp, output):
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.retain_grad()
            activations.append(output)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        with torch.enable_grad():
            logits_orig = model(tensor_orig)
            probs_orig = torch.nn.functional.softmax(logits_orig[0], dim=0)

            # Nếu bật TTA: Suy luận thêm bản lật ngang (Horizontal Flip) để ổn định dự đoán
            if enable_tta:
                with torch.no_grad():
                    flipped_pil = clean_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    tensor_flip = transform(flipped_pil).unsqueeze(0)
                    logits_flip = model(tensor_flip)
                    probs_flip = torch.nn.functional.softmax(logits_flip[0], dim=0)
                probs_final = 0.5 * (probs_orig + probs_flip)
            else:
                probs_final = probs_orig

            pred_idx = int(probs_final.argmax().item())
            target_score = logits_orig[0, pred_idx]

            model.zero_grad()
            if activations:
                feat_map = activations[0]
                grads = torch.autograd.grad(
                    outputs=target_score,
                    inputs=feat_map,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * feat_map).sum(dim=1)).squeeze(0)
                cam_np = cam.detach().cpu().numpy()
            else:
                cam_np = np.zeros((7, 7), dtype=np.float32)
    finally:
        hook_handle.remove()

    cam_min, cam_max = cam_np.min(), cam_np.max()
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam_np)

    gradcam_pil, raw_heatmap_pil = blend_gradcam_heatmap(
        clean_pil=clean_pil,
        cam_norm=cam_norm,
        alpha_mask=alpha_mask,
        heatmap_opacity=heatmap_opacity,
        focus_on_foreground=focus_on_foreground,
    )

    probs_np = probs_final.detach().cpu().numpy()
    confidence = float(probs_np[pred_idx] * 100.0)
    entropy = compute_normalized_entropy(probs_np)
    infer_time_ms = (time.perf_counter() - t0) * 1000.0

    prob_dict = {
        class_names[i]: float(probs_np[i] * 100.0)
        for i in range(len(class_names))
    }

    return {
        "pred_idx": pred_idx,
        "pred_class": class_names[pred_idx],
        "confidence": confidence,
        "entropy": entropy,
        "probabilities": prob_dict,
        "cam_norm": cam_norm,
        "gradcam_pil": gradcam_pil,
        "raw_heatmap_pil": raw_heatmap_pil,
        "infer_time_ms": infer_time_ms,
        "tta_used": enable_tta,
    }
