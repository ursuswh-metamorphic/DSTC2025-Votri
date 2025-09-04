# DSTC Project – Environment Setup Guide

## 1. Yêu cầu hệ thống
- **Hệ điều hành:** Ubuntu (khuyến nghị sử dụng WSL latest version trên Windows)  
- **Trình quản lý môi trường:** [Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html#install-linux-silent)  
- **Python:** >= 3.8  

---

## 2. Cài đặt Miniconda
### Bước 1: Tải Miniconda cho Linux
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### Bước 2: Cài đặt
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
- Chọn `yes` để đồng ý với điều khoản.  

### Bước 3: Khởi động lại Terminal và kiểm tra
```bash
conda --version
```
Nếu hiển thị phiên bản thì cài đặt thành công.

---

## 3. Tạo môi trường Conda
```bash
conda create -n dstc python=3.9
conda activate dstc
```

---

## 4. Cài đặt thư viện cần thiết
### 4.1 Cập nhật và cài các gói hệ thống
```bash
sudo apt-get update
sudo apt-get install -y pkg-config libdbus-1-dev libglib2.0-dev
```

### 4.2 Cài đặt **FiinQuantx**
```bash
pip install --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx
```

Cập nhật khi có phiên bản mới:
```bash
pip install --upgrade --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx
```

### 4.3 Cài đặt thư viện từ `requirements.txt`
```bash
pip install -r requirements.txt
```

---

## 5. Dữ liệu
Có 2 cách để lấy dữ liệu phục vụ phân tích:
1. **Dùng sẵn:** `DSTC_3year.csv`  -> file dữ lieu đã fetch sẵn bằng thư viện fiinquantx
2. **Tự crawl:** chạy script `Fetch_VN30.py`

---

## 6. Chạy chương trình
```bash
python3 <tên_file.py>

Hoặc thực hiện chạy bằng file problem2_Votri.ipynb đã có trong repo
```

---

## 7. Tài liệu tham khảo
- [Conda Documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html#install-linux-silent)  
