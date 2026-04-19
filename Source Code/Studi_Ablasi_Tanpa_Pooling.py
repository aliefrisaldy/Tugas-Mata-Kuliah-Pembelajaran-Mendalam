import numpy as np
import time
import os
import matplotlib.pyplot as plt
from Arsitektur_CNN import CNN, train, confusion_matrix

ARRAY_DIR = (
    "D:/The Journey/Semester 6/Deep Learning/"
    "Pak Yuri Yudhaswara/Pertemuan 3/"
    "Bismillah Tugas Akhir/dataset_arrays"
)
HASIL_LAMA_PATH = (
    "D:/The Journey/Semester 6/Deep Learning/"
    "Pak Yuri Yudhaswara/Pertemuan 3/"
    "Bismillah Tugas Akhir/hasil_eksperimen/hasil_eksperimen.npy"
)
OUTPUT_DIR   = "hasil_ablasi_pada_pooling_layer"
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 0.001
INPUT_SHAPE  = (64, 64, 3)
PATIENCE     = 15
LR_DECAY     = True
DECAY_EVERY  = 10
DECAY_FACTOR = 0.5


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

def load_hasil_with_pooling() -> dict:
    print(" Loading hasil eksperimen sebelumnya...")
    semua = np.load(HASIL_LAMA_PATH, allow_pickle=True)
    for h in semua:
        if h["nama"] == "3Conv-ReLU":
            hasil = {
                "nama"        : "3Conv-ReLU-WithPooling",
                "use_pooling" : True,
                "history"     : h["history"],
                "cm"          : h["cm"],
                "waktu"       : h["waktu"],
            }
            print(f"  Ditemukan: {h['nama']} → Test Acc: {h['cm']['accuracy']*100:.2f}%")
            return hasil
    raise ValueError("Model '3Conv-ReLU' tidak ditemukan di file hasil_eksperimen.npy!")


# ─────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────

def plot_akurasi(semua_hasil: list, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Kurva Akurasi Training & Testing\nAblasi Pooling Layer (3Conv-ReLU)",
                 fontsize=13, fontweight="bold")

    colors = ["steelblue", "darkorange"]

    for idx, hasil in enumerate(semua_hasil):
        ax = axes[idx]
        ep = range(1, len(hasil["history"]["train_acc"]) + 1)
        ax.plot(ep, [a * 100 for a in hasil["history"]["train_acc"]],
                label="Train", color=colors[idx], linewidth=2)
        ax.plot(ep, [a * 100 for a in hasil["history"]["test_acc"]],
                label="Test", color=colors[idx], linewidth=2, linestyle="--")
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Kurva Loss Training\nAblasi Pooling Layer (3Conv-ReLU)",
                 fontsize=13, fontweight="bold")

    colors = ["steelblue", "darkorange"]

    for idx, hasil in enumerate(semua_hasil):
        ax = axes[idx]
        ep = range(1, len(hasil["history"]["loss"]) + 1)
        ax.plot(ep, hasil["history"]["loss"], color=colors
                [idx], linewidth=2)
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Confusion Matrix per Model (Data Test)\nAblasi Pooling Layer (3Conv-ReLU)",
                 fontsize=13, fontweight="bold")

    label_names = ["Mobil", "Motor"]

    for idx, hasil in enumerate(semua_hasil):
        ax = axes[idx]
        cm = hasil["cm"]["matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(hasil["nama"], fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        fig.colorbar(im, ax=ax)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "grafik_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Tersimpan: {path}")


def plot_perbandingan(semua_hasil: list, output_dir: str):
    nama_model = [h["nama"] for h in semua_hasil]
    test_acc   = [h["cm"]["accuracy"] * 100 for h in semua_hasil]
    f1_score   = [h["cm"]["f1"] * 100 for h in semua_hasil]
    precision  = [h["cm"]["precision"] * 100 for h in semua_hasil]
    recall     = [h["cm"]["recall"] * 100 for h in semua_hasil]
    waktu      = [h["waktu"] for h in semua_hasil]

    colors = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Perbandingan: With Pooling vs Without Pooling (3Conv-ReLU)",
                 fontsize=13, fontweight="bold")

    # Plot 1: Akurasi Test
    bars = axes[0].bar(nama_model, test_acc, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Akurasi Test (%)")
    axes[0].set_ylabel("Akurasi (%)")
    axes[0].set_ylim([0, 110])
    axes[0].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, test_acc):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # Plot 2: F1, Precision, Recall
    x     = np.arange(len(nama_model))
    width = 0.25
    axes[1].bar(x - width, f1_score,  width, label="F1",       
                color="steelblue",  alpha=0.85, edgecolor="black")
    axes[1].bar(x,         precision, width, label="Precision", 
                color="darkorange", alpha=0.85, edgecolor="black")
    axes[1].bar(x + width, recall,    width, label="Recall",    
    color="seagreen",   alpha=0.85, edgecolor="black")
    axes[1].set_title("F1 / Precision / Recall (%)")
    axes[1].set_ylabel("Skor (%)")
    axes[1].set_ylim([0, 115])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["With\nPooling", "Without\nPooling"])
    axes[1].legend()

    # Plot 3: Waktu Training
    bars2 = axes[2].bar(nama_model, waktu, color=colors, edgecolor="black", alpha=0.85)
    axes[2].set_title("Waktu Training (detik)")
    axes[2].set_ylabel("Waktu (s)")
    axes[2].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, waktu):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{val:.0f}s", ha="center", va="bottom", fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue",  label="With Pooling"),
        Patch(facecolor="darkorange", label="Without Pooling"),
    ]
    axes[0].legend(handles=legend_elements)

    plt.tight_layout()
    path = os.path.join(output_dir, "grafik_perbandingan.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Tersimpan: {path}")


def simpan_tabel(semua_hasil: list, output_dir: str):
    lines = []
    lines.append("=" * 95)
    lines.append("  TABEL PERBANDINGAN HASIL ABLASI POOLING LAYER (3Conv-ReLU)")
    lines.append("=" * 95)
    lines.append(f"{'Model':<30} {'Train Acc':>10} {'Test Acc':>10} "
                 f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Best Epoch':>12} {'Waktu':>10}")
    lines.append("-" * 95)

    for h in semua_hasil:
        cm      = h["cm"]
        best_ep = h["history"].get("best_epoch", "-")
        lines.append(
            f"{h['nama']:<30} "
            f"{h['history']['train_acc'][-1] * 100:>9.2f}% "
            f"{cm['accuracy'] * 100:>9.2f}% "
            f"{cm['precision'] * 100:>9.2f}% "
            f"{cm['recall'] * 100:>7.2f}% "
            f"{cm['f1'] * 100:>7.2f}% "
            f"{str(best_ep):>12} "
            f"{h['waktu']:>8.1f}s"
        )

    lines.append("=" * 95)

    with_pool     = semua_hasil[0]
    without_pool  = semua_hasil[1]
    selisih_acc   = (with_pool["cm"]["accuracy"] - without_pool["cm"]["accuracy"]) * 100
    selisih_f1    = (with_pool["cm"]["f1"] - without_pool["cm"]["f1"]) * 100
    selisih_waktu = without_pool["waktu"] - with_pool["waktu"]

    lines.append("\n KESIMPULAN ABLASI:")
    lines.append("   With Pooling:")
    lines.append(f"      Test Acc : {with_pool['cm']['accuracy']*100:.2f}%")
    lines.append(f"      F1 Score : {with_pool['cm']['f1']*100:.2f}%")
    lines.append(f"      Waktu    : {with_pool['waktu']:.1f}s")

    lines.append("   Without Pooling:")
    lines.append(f"      Test Acc : {without_pool['cm']['accuracy']*100:.2f}%")
    lines.append(f"      F1 Score : {without_pool['cm']['f1']*100:.2f}%")
    lines.append(f"      Waktu    : {without_pool['waktu']:.1f}s")

    lines.append("   Perbandingan:")
    lines.append(f"      Selisih Akurasi  : {abs(selisih_acc):.2f}%")
    lines.append(f"         ({'With Pooling lebih baik' if selisih_acc > 0 
                              else 'Without Pooling lebih baik'})")

    lines.append(f"      Selisih F1 Score : {abs(selisih_f1):.2f}%")
    lines.append(f"         ({'With Pooling lebih baik' if selisih_f1 > 0 
                              else 'Without Pooling lebih baik'})")

    lines.append(f"      Selisih Waktu    : {abs(selisih_waktu):.1f}s")
    lines.append(f"         ({'Without Pooling lebih lambat' if selisih_waktu > 
                              0 else 'Without Pooling lebih cepat'})")
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
    print("=" * 60)
    print("  ABLASI POOLING LAYER - MODEL TERBAIK (3Conv-ReLU)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hasil_with_pooling = load_hasil_with_pooling()

    X_train, y_train, X_test, y_test = load_data()

    print(f"\n{'─'*60}")
    print(f"  Training : 3Conv-ReLU-WithoutPooling")
    print(f"  Pooling  : OFF")
    print(f"{'─'*60}")

    np.random.seed(42)
    model = CNN(
        n_conv      = 3,
        activation  = "relu",
        lr          = LR,
        use_pooling = False,
        input_shape = INPUT_SHAPE,
    )

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

    cm = confusion_matrix(model, X_test, y_test)

    print(f"\n   Hasil Akhir 3Conv-ReLU-WithoutPooling:")
    print(f"     Best Epoch : {history['best_epoch']}/{EPOCHS}")
    print(f"     Train Acc  : {history['train_acc'][history
                                                    ['best_epoch']-1]*100:.2f}%")
    print(f"     Test Acc   : {history['best_test_acc']*100:.2f}%")
    print(f"     F1 Score   : {cm['f1']*100:.2f}%")
    print(f"     Waktu      : {waktu:.1f} detik")

    hasil_without_pooling = {
        "nama"        : "3Conv-ReLU-WithoutPooling",
        "use_pooling" : False,
        "history"     : history,
        "cm"          : cm,
        "waktu"       : waktu,
    }

    semua_hasil = [hasil_with_pooling, hasil_without_pooling]

    print(f"\n{'═'*60}")
    print("  MENYIMPAN HASIL & GRAFIK")
    print(f"{'═'*60}")

    np.save(os.path.join(OUTPUT_DIR, "hasil_ablasi_pooling.npy"),
            semua_hasil, allow_pickle=True)

    plot_akurasi(semua_hasil, OUTPUT_DIR)
    plot_loss(semua_hasil, OUTPUT_DIR)
    plot_confusion_matrices(semua_hasil, OUTPUT_DIR)
    plot_perbandingan(semua_hasil, OUTPUT_DIR)
    simpan_tabel(semua_hasil, OUTPUT_DIR)

    print(f"\n Semua eksperimen ablasi selesai!")
    print(f"   Output tersimpan di folder: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()