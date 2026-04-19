# 🚗🏍️ Klasifikasi Mobil vs Motor menggunakan CNN from Scratch

> **Tugas Akhir Mata Kuliah Pembelajaran Mendalam**  
> Program Studi Informatika — Universitas Tadulako  

## 👥 Kelompok 3

| Nama | NIM |
|------|-----|
| Muhammad Alif Risaldy | F55123055 |
| Isti Zahra Eka Putri Katili | F55123029 |
| Reyhan Dany A.H. Mohammad | F55123095 |

---

## 📋 Deskripsi

Proyek ini mengimplementasikan **Convolutional Neural Network (CNN) dari nol** menggunakan Python dan NumPy **tanpa framework deep learning** seperti PyTorch, Keras, maupun TensorFlow. Model digunakan untuk melakukan **klasifikasi biner** antara dua kelas kendaraan yaitu **mobil** dan **motor**.

Eksperimen dirancang dalam bentuk **studi ablasi** dengan membandingkan 6 kombinasi model berdasarkan:
- Jumlah convolutional layer: **1, 2, dan 3 layer**
- Fungsi aktivasi: **ReLU dan Sigmoid**

---

## 🏗️ Arsitektur CNN

Setiap model terdiri dari susunan layer berikut:

```
Input (64×64×3)
    ↓
[Conv Layer → Activation Layer → Max Pooling] × n
    ↓
Flatten
    ↓
Fully Connected (64 neuron) → Activation Layer
    ↓
Fully Connected (1 neuron) → Sigmoid
    ↓
Output (0=Mobil, 1=Motor)
```

---

## 📁 Struktur Repository

```
Tugas_MKPM_Kelompok3/
│
├── Source Code/
│     ├── Arsitektur_CNN.py              ← Implementasi CNN (semua komponen)
│     ├── Scraping.py                    ← Scraping dataset dari Google & Bing
│     ├── Preprocessing.py               ← Resize + split train/test + simpan .npy
│     ├── Augmentasi.py                  ← Augmentasi data training
│     ├── Train_6_Kombinasi_Model.py     ← Training 6 kombinasi model + evaluasi
│     └── Studi_Ablasi_Tanpa_Pooling.py  ← Studi ablasi pengaruh pooling layer
│
├── Demo Aplikasi/
│     ├── Demo.py                        ← Aplikasi prediksi gambar
│     └── hasil_prediksi/
│           ├── hasil_prediksi_1.png
│           ├── hasil_prediksi_2.png
│           ├── hasil_prediksi_3.png
│           └── hasil_prediksi_4.png
│
├── Model Terbaik/
│     ├── bobot_3conv_relu.npy           ← Bobot model terbaik
│     └── config_model.npy              ← Konfigurasi model
│
├── Dataset/
│     ├── mobil/                         ← 500 gambar mobil
│     └── motor/                         ← 500 gambar motor
│
├── Hasil Eksperimen/
│     ├── grafik_akurasi.png
│     ├── grafik_loss.png
│     ├── grafik_confusion.png
│     ├── grafik_perbandingan.png
│     └── tabel_perbandingan.txt
│
├── Laporan dan Slide Presentasi/
│     ├── Laporan.pdf
│     └── Slide.pptx
│
└── README.md
```

---

## 🔧 Requirements

```bash
pip install numpy Pillow matplotlib icrawler
```

| Library | Kegunaan |
|---------|----------|
| numpy | Implementasi CNN (wajib) |
| Pillow | Pemrosesan gambar |
| matplotlib | Visualisasi hasil |
| icrawler | Scraping dataset |
| tkinter | File dialog demo (bawaan Python) |

---

## ▶️ Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/Tugas_MKPM_Kelompok3.git](https://github.com/aliefrisaldy/Tugas-Mata-Kuliah-Pembelajaran-Mendalam
cd Tugas_MKPM_Kelompok3/source_code
```

### 2. Install Dependencies
```bash
pip install numpy Pillow matplotlib icrawler
```

### 3. Jalankan Pipeline Secara Berurutan

```bash
# Step 1 - Scraping dataset (opsional, dataset sudah tersedia)
python Scraping.py

# Step 2 - Preprocessing: resize + split 70/30
python Preprocessing.py

# Step 3 - Augmentasi data training (6x lipat)
python Augmentasi.py

# Step 4 - Training 6 kombinasi model + simpan hasil
python Train_6_Kombinasi_Model.py
```

### 4. Jalankan Studi Ablasi Tambahan (Opsional)
```bash
python Studi_Ablasi_Tanpa_Pooling.py
```

### 5. Jalankan Demo Aplikasi
```bash
cd ../Demo\ Aplikasi
python Demo.py
```

---

## 📊 Hasil Eksperimen

| Model | Train Acc | Test Acc | Precision | Recall | F1 | Waktu |
|-------|-----------|----------|-----------|--------|----|-------|
| 1Conv-ReLU | 80.88% | 72.67% | 68.68% | 83.33% | 75.30% | 889.7s |
| 2Conv-ReLU | 80.74% | 77.33% | 74.40% | 83.33% | 78.62% | 1963.4s |
| **3Conv-ReLU** | **77.48%** | **79.00%** | **80.85%** | **76.00%** | **78.35%** | **2775.2s** |
| 1Conv-Sigmoid | 60.33% | 60.33% | 63.96% | 47.33% | 54.41% | 1463.2s |
| 2Conv-Sigmoid | 59.57% | 61.67% | 60.12% | 69.33% | 64.40% | 3886.8s |
| 3Conv-Sigmoid | 52.26% | 57.33% | 56.25% | 66.00% | 60.74% | 1835.5s |

**🏆 Model Terbaik: 3Conv-ReLU (Test Accuracy: 79.00%)**

---

## 🔍 Temuan Utama

- **ReLU vs Sigmoid**: ReLU secara konsisten mengungguli Sigmoid pada seluruh metrik evaluasi karena kemampuannya mengatasi vanishing gradient
- **Jumlah Layer**: Penambahan convolutional layer meningkatkan performa pada ReLU namun dengan trade-off waktu training yang lebih lama
- **Augmentasi**: Teknik flip horizontal, rotasi ±15°, dan perubahan kecerahan berhasil meningkatkan kemampuan generalisasi model

---

## 🎮 Demo Aplikasi

Model terbaik (3Conv-ReLU) dapat digunakan langsung untuk memprediksi gambar baru melalui demo aplikasi. Pengguna cukup memilih gambar melalui file dialog dan hasil prediksi beserta confidence score ditampilkan secara visual.

```
📁 Pilih gambar → Model prediksi → Tampilkan hasil + confidence
```

---

## 📝 Catatan

- Seluruh komponen CNN (forward pass, backward pass, backpropagation) diimplementasikan **manual menggunakan NumPy**
- Training menggunakan CPU sehingga waktu training relatif lama
- Dataset diperoleh melalui scraping dari Google Images dan Bing menggunakan icrawler
