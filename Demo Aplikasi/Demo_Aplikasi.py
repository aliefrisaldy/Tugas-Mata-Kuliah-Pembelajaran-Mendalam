import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
from Arsitektur_CNN import CNN


MODEL_DIR   = "D:\The Journey\Semester 6\Deep Learning\Pak Yuri Yudhaswara\Pertemuan 9\Bismillah Tugas Akhir\Model Terbaik"
BOBOT_PATH  = os.path.join(MODEL_DIR, "bobot_3conv_relu.npy")
CONFIG_PATH = os.path.join(MODEL_DIR, "config_model.npy")
IMG_SIZE    = (64, 64)
LABEL       = {0: "MOBIL", 1: "MOTOR"}
WARNA       = {0: "#2196F3", 1: "#FF5722"}


# LOAD MODEL


def load_model():
    """Load model dari file bobot yang tersimpan."""
    if not os.path.exists(BOBOT_PATH):
        raise FileNotFoundError(
            f"File bobot tidak ditemukan: {BOBOT_PATH}\n"
        )

    config = np.load(CONFIG_PATH, allow_pickle=True).item()
    model  = CNN(
        n_conv      = config["n_conv"],
        activation  = config["activation"],
        use_pooling = config["use_pooling"],
        input_shape = config["input_shape"],
    )
    bobot = np.load(BOBOT_PATH, allow_pickle=True).tolist()
    model.set_weights(bobot)
    return model, config


# ─────────────────────────────────────────
# UPLOAD GAMBAR VIA FILE DIALOG
# ─────────────────────────────────────────

def upload_gambar() -> str:
    """Buka file dialog OS untuk memilih gambar."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title     = "Pilih Gambar (Mobil atau Motor)",
        filetypes = [
            ("File Gambar", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("Semua File",  "*.*"),
        ]
    )
    root.destroy()
    return path


# ─────────────────────────────────────────
# PREPROCESS & PREDIKSI
# ─────────────────────────────────────────

def preprocess_gambar(path: str) -> np.ndarray:
    """Load gambar, resize, normalisasi."""
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, *arr.shape)


def prediksi(model: CNN, img_arr: np.ndarray):
    """Jalankan prediksi dan return label + probabilitas."""
    pred, prob = model.predict(img_arr)
    label_idx  = int(pred[0])
    prob_motor = float(prob[0])
    prob_mobil = 1 - prob_motor
    confidence = prob_motor if label_idx == 1 else prob_mobil
    return label_idx, confidence, prob_mobil, prob_motor


# ─────────────────────────────────────────
# TAMPILKAN HASIL
# ─────────────────────────────────────────

def tampilkan_hasil(path: str, label_idx: int, confidence: float,
                    prob_mobil: float, prob_motor: float):
    """Tampilkan gambar + hasil prediksi + bar confidence."""

    img_asli = Image.open(path).convert("RGB")
    warna    = WARNA[label_idx]
    label    = LABEL[label_idx]

    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor("#1E1E1E")
    fig.suptitle("DEMO KLASIFIKASI: MOBIL vs MOTOR",
                 fontsize=14, fontweight="bold",
                 color="white", y=0.98)

    ax1 = fig.add_axes([0.03, 0.08, 0.45, 0.82])
    ax1.imshow(img_asli)
    ax1.set_title(os.path.basename(path),
                  color="white", fontsize=9, pad=6)
    ax1.axis("off")

    for spine in ax1.spines.values():
        spine.set_edgecolor(warna)
        spine.set_linewidth(3)
        spine.set_visible(True)

    ax2 = fig.add_axes([0.52, 0.08, 0.45, 0.82])
    ax2.set_facecolor("#2D2D2D")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Label prediksi
    ax2.text(0.5, 0.88, "PREDIKSI",
             ha="center", va="center",
             color="#AAAAAA", fontsize=10)
    ax2.text(0.5, 0.73, label,
             ha="center", va="center",
             color=warna, fontsize=28, fontweight="bold")

    # Confidence
    ax2.text(0.5, 0.58, f"Confidence: {confidence*100:.2f}%",
             ha="center", va="center",
             color="white", fontsize=13)

    # Bar probabilitas Mobil
    ax2.text(0.08, 0.44, "Mobil", color="#2196F3",
             fontsize=10, va="center")
    ax2.barh(0.38, prob_mobil, height=0.07, left=0.22,
             color="#2196F3", alpha=0.85)
    ax2.text(0.22 + prob_mobil + 0.02, 0.38,
             f"{prob_mobil*100:.1f}%",
             color="white", fontsize=9, va="center")

    # Bar probabilitas Motor
    ax2.text(0.08, 0.26, "Motor", color="#FF5722",
             fontsize=10, va="center")
    ax2.barh(0.20, prob_motor, height=0.07, left=0.22,
             color="#FF5722", alpha=0.85)
    ax2.text(0.22 + prob_motor + 0.02, 0.20,
             f"{prob_motor*100:.1f}%",
             color="white", fontsize=9, va="center")

    # Garis batas bar
    ax2.axvline(x=0.22, ymin=0.13, ymax=0.52,
                color="#555555", linewidth=1)

    # Info model
    ax2.text(0.5, 0.08,
             "Model: 3Conv-ReLU | CNN from Scratch (NumPy)",
             ha="center", va="center",
             color="#777777", fontsize=8)

    plt.savefig("hasil_prediksi.png", dpi=150,
                bbox_inches="tight", facecolor="#1E1E1E")
    plt.show()
    print(f"   Hasil disimpan: hasil_prediksi.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 50)
    print("  DEMO KLASIFIKASI: MOBIL vs MOTOR")
    print("  Model: 3Conv-ReLU (CNN from scratch)")
    print("=" * 50)

    # Load model
    print("\n Loading model...")
    try:
        model, config = load_model()
        print(f"   Model berhasil dimuat!")
        print(f"  Arsitektur : {config['n_conv']} Conv Layer + "
              f"{config['activation'].upper()}")
        print(f"  Test Acc   : {config['test_acc']*100:.2f}%")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        return

    # Loop prediksi
    while True:
        print("\n" + "─" * 50)
        print("  1. Pilih gambar baru")
        print("  2. Keluar")
        pilihan = input("\n  Pilihan (1/2): ").strip()

        if pilihan == "2":
            print("\n Terima kasih! Program selesai.")
            break

        if pilihan != "1":
            continue

        # Upload gambar via file dialog
        print("\n Membuka file dialog...")
        path = upload_gambar()

        if not path:
            print("    Tidak ada gambar dipilih.")
            continue

        print(f"   Gambar dipilih: {os.path.basename(path)}")

        try:
            # Preprocess
            img_arr = preprocess_gambar(path)

            # Prediksi
            label_idx, confidence, prob_mobil, prob_motor = prediksi(
                model, img_arr
            )

            # Print ke terminal
            print(f"\n  Prediksi   : {LABEL[label_idx]}")
            print(f"  Confidence : {confidence*100:.2f}%")
            print(f"  Prob Mobil : {prob_mobil*100:.2f}%")
            print(f"  Prob Motor : {prob_motor*100:.2f}%")

            # Tampilkan visual
            tampilkan_hasil(path, label_idx, confidence,
                            prob_mobil, prob_motor)

        except Exception as e:
            print(f"   Gagal memproses gambar: {e}")


if __name__ == "__main__":
    main()