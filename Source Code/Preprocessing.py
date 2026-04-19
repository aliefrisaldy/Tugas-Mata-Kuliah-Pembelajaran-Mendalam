import os
import random
import numpy as np
from PIL import Image

INPUT_DIR = (
    "D:/The Journey/Semester 6/Deep Learning/"
    "PakYuri Yudhaswara/Pertemuan 3/"
    "Bismillah Tugas Akhir/Dataset"
)
OUTPUT_DIR = (
    "D:/The Journey/Semester 6/Deep Learning/"
    "PakYuri Yudhaswara/Pertemuan 3/"
    "Bismillah Tugas Akhir/Dataset_Ready"
)
IMG_SIZE = (64, 64)
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
KELAS = ["mobil", "motor"]

def load_and_resize(path: str, size: tuple) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Gagal baca {path}: {e}")
        return None


def process_split(file_list: list, kelas: str, split: str):
    out_dir = os.path.join(OUTPUT_DIR, split, kelas)
    os.makedirs(out_dir, exist_ok=True)

    berhasil = 0
    for i, fpath in enumerate(file_list, 1):
        arr = load_and_resize(fpath, IMG_SIZE)
        if arr is None:
            continue
        ext = os.path.splitext(fpath)[1].lower() or ".jpg"
        out_path = os.path.join(out_dir, f"{kelas}_{split}_{i:04d}{ext}")
        Image.fromarray(arr).save(out_path)
        berhasil += 1

    return berhasil




def save_as_numpy():
    LABEL = {"mobil": 0, "motor": 1}
    VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}
    os.makedirs("dataset_arrays", exist_ok=True)

    for split in ["train", "test"]:
        X, y = [], []
        for kelas, label in LABEL.items():
            folder = os.path.join(OUTPUT_DIR, split, kelas)
            if not os.path.exists(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if os.path.splitext(fname)[1].lower() not in VALID_EXT:
                    continue
                arr = load_and_resize(os.path.join(folder, fname), IMG_SIZE)
                if arr is not None:
                    X.append(arr)
                    y.append(label)

        if X:
            X_arr = np.array(X, dtype=np.float32) / 255.0
            y_arr = np.array(y, dtype=np.int32)
            np.save(f"dataset_arrays/X_{split}.npy", X_arr)
            np.save(f"dataset_arrays/y_{split}.npy", y_arr)
            print(f"Saved X_{split}.npy {X_arr.shape} | y_{split}.npy {y_arr.shape}")

def main():
    random.seed(RANDOM_SEED)

    print("=" * 50)
    print("PREPROCESSING RESIZE SPLIT DATASET")
    print("=" * 50)

    VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}
    ringkasan = {}

    for kelas in KELAS:
        src_dir = os.path.join(INPUT_DIR, kelas)
        if not os.path.exists(src_dir):
            print(f"Folder tidak ditemukan: {src_dir}")
            continue

        semua_file = [
            os.path.join(src_dir, f)
            for f in os.listdir(src_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXT
        ]
        random.shuffle(semua_file)

        cut = int(len(semua_file) * TRAIN_RATIO)
        train_list = semua_file[:cut]
        test_list = semua_file[cut:]

        print(f"Kelas: {kelas.upper()}")
        print(f"Total: {len(semua_file)}")
        print(f"Train: {len(train_list)}")
        print(f"Test: {len(test_list)}")

        n_train = process_split(train_list, kelas, "train")
        n_test = process_split(test_list, kelas, "test")

        ringkasan[kelas] = {"train": n_train, "test": n_test}
        print(f"Tersimpan train: {n_train}, test: {n_test}")

    print("Membuat numpy arrays")
    save_as_numpy()

    print("=" * 50)
    print("RINGKASAN AKHIR")
    print("=" * 50)
    for kelas, info in ringkasan.items():
        print(f"{kelas:10s} train: {info['train']:4d} | test: {info['test']:4d}")
    print(f"Ukuran gambar: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Output folder: {OUTPUT_DIR}/")
    print("Preprocessing selesai")

if __name__ == "__main__":
    main()