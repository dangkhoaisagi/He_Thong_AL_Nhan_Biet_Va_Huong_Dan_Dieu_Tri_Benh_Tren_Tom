# AQUA-XAI Pro - Hệ thống AI hỗ trợ chẩn đoán bệnh tôm

Ứng dụng web Streamlit hỗ trợ phân tích ảnh tôm bằng ConvNeXt-Tiny, tách nền U2-Net và giải thích dự đoán bằng Grad-CAM. Ứng dụng có các chức năng phân tích ảnh đơn/lô, đánh giá chất lượng ảnh và độ tin cậy, tra cứu thông tin bệnh, lưu lịch sử cục bộ và xuất báo cáo.

> **Lưu ý:** Đây là công cụ hỗ trợ tham khảo, không thay thế kết quả xét nghiệm hoặc tư vấn của cán bộ thú y thủy sản. Hãy xác minh kết quả trước khi quyết định điều trị.

## Tính năng

- Phân loại ảnh theo các lớp `Healthy`, `WSSV`, `BG` và `BG_WSSV`.
- Tách nền tôm bằng U2-Net (thông qua `rembg`) và tùy chọn cắt vùng tôm.
- Hiển thị Grad-CAM, xác suất dự đoán, độ bất định và các cảnh báo chất lượng ảnh.
- Hỗ trợ suy luận TTA bằng ảnh lật ngang và phân tích nhiều ảnh.
- Tra cứu kiến thức bệnh học, gợi ý xử lý và xuất báo cáo.
- Lưu lịch sử chẩn đoán trên máy đang chạy ứng dụng trong `diagnosis_history.json`.

## Yêu cầu

- Windows 10/11, macOS hoặc Linux.
- Python 3.10–3.12 khuyến nghị.
- Git và Git LFS để tải checkpoint mô hình được lưu trong GitHub LFS.
- Kết nối Internet ở lần chạy đầu để tải model U2-Net mà `rembg` sử dụng.
- Máy có RAM/GPU phù hợp với PyTorch. Có thể chạy CPU; GPU NVIDIA cần cài phiên bản PyTorch/CUDA tương thích riêng.

Checkpoint `best_shrimp_convnext.pth` được quản lý bằng Git LFS. Sau khi clone, cần tải file LFS về thì ứng dụng mới chạy được.

## Cài đặt và chạy trên Windows

Mở PowerShell hoặc Command Prompt:

1. Cài Python, Git và Git LFS nếu máy chưa có. Sau khi cài Git LFS, khởi tạo cấu hình:

   ```powershell
   git lfs install
   ```

2. Clone dự án và tải các file LFS:

   ```powershell
   git clone https://github.com/dangkhoaisagi/He_Thong_AL_Nhan_Biet_Va_Huong_Dan_Dieu_Tri_Benh_Tren_Tom.git
   cd He_Thong_AL_Nhan_Biet_Va_Huong_Dan_Dieu_Tri_Benh_Tren_Tom
   git lfs pull
   ```

3. Tạo môi trường ảo và cài dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Nếu PowerShell không nhận lệnh `py`, cài Python và dùng `python -m venv .venv` thay thế. Nếu PowerShell chặn kích hoạt môi trường ảo, có thể chạy lệnh sau trong cửa sổ hiện tại rồi kích hoạt lại:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

4. Chạy ứng dụng:

   ```powershell
   python -m streamlit run app.py
   ```

   Hoặc nhấp đúp `run_app.bat` sau khi đã cài dependencies và đảm bảo Python trong môi trường ảo đã được kích hoạt trong terminal. Streamlit thường mở trình duyệt tại [http://localhost:8501](http://localhost:8501).

## Chạy trên macOS/Linux

```bash
git lfs install
git clone https://github.com/dangkhoaisagi/He_Thong_AL_Nhan_Biet_Va_Huong_Dan_Dieu_Tri_Benh_Tren_Tom.git
cd He_Thong_AL_Nhan_Biet_Va_Huong_Dan_Dieu_Tri_Benh_Tren_Tom
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Sau đó mở [http://localhost:8501](http://localhost:8501) nếu trình duyệt không tự mở.

## Chạy nhanh trên Windows bằng file BAT

Sau khi clone, tải LFS, cài dependencies và kích hoạt đúng môi trường Python, chạy:

```bat
run_app.bat
```

Hoặc dùng lệnh trực tiếp từ thư mục dự án:

```bat
python -m streamlit run app.py
```

## Tệp và dữ liệu cần lưu ý

- `best_shrimp_convnext.pth`: trọng số ConvNeXt-Tiny, được lưu trong Git LFS.
- `diagnosis_history.json`: dữ liệu lịch sử được tạo/cập nhật cục bộ; không nên đưa dữ liệu chẩn đoán cá nhân lên GitHub.
- `assets/`: hình ảnh minh họa giao diện.
- Các file `.py` ở thư mục gốc: mã nguồn ứng dụng, xử lý XAI và cơ sở tri thức.

Nếu ứng dụng báo không tìm thấy checkpoint, kiểm tra file `best_shrimp_convnext.pth` có nằm cạnh `app.py` và đã được tải đầy đủ bằng `git lfs pull` hay chưa. Nếu file chứa nội dung bắt đầu bằng `version https://git-lfs.github.com/spec/v1`, Git LFS chưa tải dữ liệu model thực tế.

## Cấu trúc dự án

```text
.
├── app.py
├── knowledge_base.py
├── xai_utils.py
├── ui_theme.py
├── best_shrimp_convnext.pth
├── requirements.txt
├── run_app.bat
├── assets/
└── README.md
```

## Giấy phép và đóng góp

Chưa khai báo giấy phép sử dụng. Vui lòng liên hệ chủ dự án trước khi tái sử dụng mã nguồn, trọng số mô hình hoặc nội dung cơ sở tri thức.
