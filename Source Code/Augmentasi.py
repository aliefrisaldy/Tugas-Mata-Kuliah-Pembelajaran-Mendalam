import numpy as np
import os
from PIL import Image, ImageEnhance

INPUT_DIR = "dataset_arrays"
OUTPUT_DIR = "dataset_arrays"

AUGMENTASI = {
    "flip_horizontal": True,
    "rotasi": True,
    "brightness": True,
}

SUDUT_ROTASI = [-15, 15]
BRIGHTNESS_FACTOR = [0.7, 1.3]


def flip_horizontal(img_arr: np.ndarray) -> np.ndarray:
    return img_arr[:, ::-1, :]


def rotasi(img_arr: np.ndarray, sudut: float) -> np.ndarray:
    img_uint8 = (img_arr * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_rot = img_pil.rotate(sudut, resample=Image.BILINEAR, expand=False)
    return np.array(img_rot, dtype=np.float32) / 255.0


def ubah_brightness(img_arr: np.ndarray, factor: float) -> np.ndarray:
    img_uint8 = (img_arr * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    enhancer = ImageEnhance.Brightness(img_pil)
    img_bright = enhancer.enhance(factor)
    return np.array(img_bright, dtype=np.float32) / 255.0


def augmentasi_satu_gambar(img_arr: np.ndarray) -> list:
    hasil = []

    if AUGMENTASI["flip_horizontal"]:
        hasil.append(flip_horizontal(img_arr))

    if AUGMENTASI["rotasi"]:
        for sudut in SUDUT_ROTASI:
            hasil.append(rotasi(img_arr, sudut))

    if AUGMENTASI["brightness"]:
        for factor in BRIGHTNESS_FACTOR:
            hasil.append(ubah_brightness(img_arr, factor))

    return hasil


def hitung_multiplier() -> int:
    n = 1
    if AUGMENTASI["flip_horizontal"]:
        n += 1
    if AUGMENTASI["rotasi"]:
        n += len(SUDUT_ROTASI)
    if AUGMENTASI["brightness"]:
        n += len(BRIGHTNESS_FACTOR)
    return n


def main():
    print("=" * 55)
    print("AUGMENTASI DATA TRAINING")
    print("=" * 55)

    x_path = os.path.join(INPUT_DIR, "X_train.npy")
    y_path = os.path.join(INPUT_DIR, "y_train.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("File X_train.npy / y_train.npy tidak ditemukan")
        return

    X_train = np.load(x_path).astype(np.float32)
    y_train = np.load(y_path)

    N = len(X_train)
    multiplier = hitung_multiplier()

    print(f"Data original: {N}")
    aktif = [k for k, v in AUGMENTASI.items() if v]
    print("Teknik:", ", ".join(aktif))
    print(f"Estimasi: {N} x {multiplier} = {N * multiplier}")

    X_aug_list = []
    y_aug_list = []

    for i, (img, label) in enumerate(zip(X_train, y_train)):
        X_aug_list.append(img)
        y_aug_list.append(label)

        for img_augmented in augmentasi_satu_gambar(img):
            X_aug_list.append(img_augmented)
            y_aug_list.append(label)

        if (i + 1) % 50 == 0 or (i + 1) == N:
            print(f"{i+1}/{N} → {len(X_aug_list)}")

    X_aug = np.array(X_aug_list, dtype=np.float32)
    y_aug = np.array(y_aug_list, dtype=np.int32)

    print("Shuffle data")
    idx = np.random.permutation(len(X_aug))
    X_aug = X_aug[idx]
    y_aug = y_aug[idx]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x_out = os.path.join(OUTPUT_DIR, "X_train_aug.npy")
    y_out = os.path.join(OUTPUT_DIR, "y_train_aug.npy")

    np.save(x_out, X_aug)
    np.save(y_out, y_aug)

    print("=" * 55)
    print("RINGKASAN")
    print("=" * 55)
    print(f"Sebelum: {N} → {X_train.shape}")
    print(f"Sesudah: {len(X_aug)} → {X_aug.shape}")

    label_names = {0: "mobil", 1: "motor"}
    for label_val, label_name in label_names.items():
        jumlah = np.sum(y_aug == label_val)
        print(f"{label_name}: {jumlah}")

    print("File:")
    print(x_out)
    print(y_out)
    print("Selesai")


if __name__ == "__main__":
    main()