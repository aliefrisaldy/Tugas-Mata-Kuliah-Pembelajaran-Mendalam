
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Arsitektur_CNN import CNN, train, confusion_matrix

ARRAY_DIR = (
    r"D:\The Journey\Semester 6\Deep Learning\Pak Yuri Yudhaswara\Pertemuan 3\Bismillah Tugas Akhir\dataset_arrays"
)
OUTPUT_DIR  = "hasil_eksperimen"
EPOCHS        = 50        
BATCH_SIZE    = 32
LR            = 0.001     
USE_POOLING   = True
INPUT_SHAPE   = (64, 64, 3)
PATIENCE      = 15         
LR_DECAY      = True     
DECAY_EVERY   = 10        
DECAY_FACTOR  = 0.5       

# 6 kombinasi model 
EKSPERIMEN = [
    {"nama": "1Conv-ReLU",    "n_conv": 1, "activation": "relu"},
    {"nama": "2Conv-ReLU",    "n_conv": 2, "activation": "relu"},
    {"nama": "3Conv-ReLU",    "n_conv": 3, "activation": "relu"},
    {"nama": "1Conv-Sigmoid", "n_conv": 1, "activation": "sigmoid"},
    {"nama": "2Conv-Sigmoid", "n_conv": 2, "activation": "sigmoid"},
    {"nama": "3Conv-Sigmoid", "n_conv": 3, "activation": "sigmoid"},
]


# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
def load_data():
    print(" Loading dataset...")
    X_train = np.load(os.path.join(ARRAY_DIR, "X_train_aug.npy"))
    y_train = np.load(os.path.join(ARRAY_DIR, "y_train_aug.npy"))
    X_test  = np.load(os.path.join(ARRAY_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(ARRAY_DIR, "y_test.npy"))

    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────

def plot_akurasi(semua_hasil: list, output_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Kurva Akurasi Training & Testing per Model", 
                 fontsize=14, fontweight="bold")

    for idx, hasil in enumerate(semua_hasil):
        ax    = axes[idx // 3][idx % 3]
        ep    = range(1, len(hasil["history"]["train_acc"]) + 1)
        ax.plot(ep, [a*100 for a in hasil["history"]["train_acc"]],
                label="Train", color="steelblue", linewidth=2)
        ax.plot(ep, [a*100 for a in hasil["history"]["test_acc"]],
                label="Test",  color="tomato",    linewidth=2, linestyle="--")
        ax.set_title(hasil["nama"], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Akurasi (%)")
        ax.legend()
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "grafik_akurasi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Tersimpan: {path}")


def plot_loss(semua_hasil: list, output_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Kurva Loss Training per Model", 
                 fontsize=14, fontweight="bold")

    for idx, hasil in enumerate(semua_hasil):
        ax = axes[idx // 3][idx % 3]
        ep = range(1, len(hasil["history"]["loss"]) + 1)
        ax.plot(ep, hasil["history"]["loss"], color="darkorange", linewidth=2)
        ax.set_title(hasil["nama"], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "grafik_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Tersimpan: {path}")


def plot_confusion_matrices(semua_hasil: list, output_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Confusion Matrix per Model (Data Test)", 
                 fontsize=14, fontweight="bold")

    label_names = ["Mobil", "Motor"]

    for idx, hasil in enumerate(semua_hasil):
        ax  = axes[idx // 3][idx % 3]
        cm  = hasil["cm"]["matrix"]
        im  = ax.imshow(cm, cmap="Blues")
        ax.set_title(hasil["nama"], fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        fig.colorbar(im, ax=ax)

        # Tambahkan angka di tiap sel
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black",
                        fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "grafik_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Tersimpan: {path}")


def plot_perbandingan(semua_hasil: list, output_dir: str):
    nama_model  = [h["nama"] for h in semua_hasil]
    test_acc    = [h["cm"]["accuracy"] * 100 for h in semua_hasil]
    waktu       = [h["waktu"] for h in semua_hasil]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Perbandingan Antar Model", fontsize=14, fontweight="bold")

    # Akurasi
    colors = ["steelblue" if "ReLU" in n else "tomato" for n in nama_model]
    bars   = ax1.bar(nama_model, test_acc, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Akurasi Test (%)")
    ax1.set_ylabel("Akurasi (%)")
    ax1.set_ylim([0, 110])
    ax1.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, test_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # Waktu training
    bars2 = ax2.bar(nama_model, waktu, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title("Waktu Training (detik)")
    ax2.set_ylabel("Waktu (s)")
    ax2.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars2, waktu):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}s", ha="center", va="bottom", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="ReLU"),
        Patch(facecolor="tomato",    label="Sigmoid"),
    ]
    ax1.legend(handles=legend_elements)

    plt.tight_layout()
    path = os.path.join(output_dir, "grafik_perbandingan.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Tersimpan: {path}")


def simpan_tabel(semua_hasil: list, output_dir: str):
    lines = []
    lines.append("=" * 85)
    lines.append("  TABEL PERBANDINGAN HASIL TRAINING 6 KOMBINASI MODEL CNN")
    lines.append("=" * 85)
    lines.append(f"{'Model':<18} {'Train Acc':>10} {'Test Acc':>10} "
                 f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Waktu':>10}")
    lines.append("-" * 85)

    for h in semua_hasil:
        cm = h["cm"]
        lines.append(
            f"{h['nama']:<18} "
            f"{h['history']['train_acc'][-1]*100:>9.2f}% "
            f"{cm['accuracy']*100:>9.2f}% "
            f"{cm['precision']*100:>9.2f}% "
            f"{cm['recall']*100:>7.2f}% "
            f"{cm['f1']*100:>7.2f}% "
            f"{h['waktu']:>8.1f}s"
        )

    lines.append("=" * 85)

    # Temukan model terbaik
    best = max(semua_hasil, key=lambda h: h["cm"]["accuracy"])
    lines.append(f"\n Model terbaik (Test Accuracy): {best['nama']} "
                 f"({best['cm']['accuracy']*100:.2f}%)")

    teks = "\n".join(lines)
    print("\n" + teks)

    path = os.path.join(output_dir, "tabel_perbandingan.txt")
    with open(path, "w") as f:
        f.write(teks)
    print(f"\n   Tersimpan: {path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 55)
    print("  TRAINING 6 KOMBINASI MODEL CNN")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    semua_hasil = []

    for i, eksp in enumerate(EKSPERIMEN, 1):
        print(f"\n{'─'*55}")
        print(f"  Model {i}/6 : {eksp['nama']}")
        print(f"  Conv layers : {eksp['n_conv']}  |  Activation : {eksp['activation']}")
        print(f"{'─'*55}")

        # Inisialisasi model baru
        np.random.seed(42)  # reproducibility
        model = CNN(
            n_conv      = eksp["n_conv"],
            activation  = eksp["activation"],
            lr          = LR,
            use_pooling = USE_POOLING,
            input_shape = INPUT_SHAPE,
        )

        # Training
        t_start = time.time()
        history = train(
            model        = model,
            X_train      = X_train,
            y_train      = y_train,
            X_test       = X_test,
            y_test       = y_test,
            epochs       = EPOCHS,
            batch_size   = BATCH_SIZE,
            patience     = PATIENCE,
            lr_decay     = LR_DECAY,
            decay_every  = DECAY_EVERY,
            decay_factor = DECAY_FACTOR,
            verbose      = True,
        )
        waktu = time.time() - t_start

        # Evaluasi
        cm = confusion_matrix(model, X_test, y_test)

        print(f"\n   Hasil Akhir {eksp['nama']}:")
        print(f"     Best Epoch : {history['best_epoch']}/{EPOCHS}")
        print(f"     Train Acc  : {history['train_acc']
                                   [history['best_epoch']-1]*100:.2f}%")
        print(f"     Test Acc   : {history['best_test_acc']*100:.2f}%")
        print(f"     F1 Score   : {cm['f1']*100:.2f}%")
        print(f"     Waktu      : {waktu:.1f} detik")

        semua_hasil.append({
            "nama"    : eksp["nama"],
            "n_conv"  : eksp["n_conv"],
            "aktiv"   : eksp["activation"],
            "history" : history,
            "cm"      : cm,
            "waktu"   : waktu,
            "model"   : model,
        })

    print(f"\n{'═'*55}")
    print("  MENYIMPAN HASIL & GRAFIK")
    print(f"{'═'*55}")

    np.save(os.path.join(OUTPUT_DIR, "hasil_eksperimen.npy"),
            semua_hasil, allow_pickle=True)

    plot_akurasi(semua_hasil, OUTPUT_DIR)
    plot_loss(semua_hasil, OUTPUT_DIR)
    plot_confusion_matrices(semua_hasil, OUTPUT_DIR)
    plot_perbandingan(semua_hasil, OUTPUT_DIR)
    simpan_tabel(semua_hasil, OUTPUT_DIR)
    # Simpan model terbaik
    best      = max(semua_hasil, key=lambda h: h["cm"]["accuracy"])
    MODEL_DIR = "model_terbaik"
    os.makedirs(MODEL_DIR, exist_ok=True)
    bobot      = best["model"].get_weights()
    bobot_path = os.path.join(MODEL_DIR,
             f"bobot_{best['nama'].lower().replace('-','_')}.npy")
    np.save(bobot_path, bobot, allow_pickle=True)
    config = {
        "nama"        : best["nama"],
        "n_conv"      : best["n_conv"],
        "activation"  : best["aktiv"],
        "use_pooling" : USE_POOLING,
        "input_shape" : INPUT_SHAPE,
        "test_acc"    : best["cm"]["accuracy"],
        "f1"          : best["cm"]["f1"],
    }
    np.save(os.path.join(MODEL_DIR, "config_model.npy"),
        config, allow_pickle=True)
    print("Menyimpan Model Terbaik")
    print(f"   Bobot tersimpan    : {bobot_path}")
    print(f"   Konfigurasi simpan : {os.path.join(MODEL_DIR, 'config_model.npy')}")
    print(f"\n Semua eksperimen selesai!")
    print(f"   Output tersimpan di folder: {OUTPUT_DIR}/")
if __name__ == "__main__":
    main()