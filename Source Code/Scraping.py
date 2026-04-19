import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

JUMLAH_GAMBAR = 500
OUTPUT_DIR = (
    "D:/The Journey/Semester 6/Deep Learning/"
    "PakYuri Yudhaswara/Pertemuan 3/"
    "Bismillah Tugas Akhir/Dataset"
)

KELAS = {
    "mobil": [
        "mobil sedan", "mobil SUV", "mobil pickup",
        "car vehicle road", "car side view", "car front view",
        "car in traffic", "car parking lot"
    ],
    "motor": [
        "sepeda motor", "motor sport", "motor matic",
        "motorcycle road", "motorcycle side view",
        "motorcycle traffic", "motorbike parking"
    ],
}

def scrape_kelas(nama_kelas: str, keyword_list: list, jumlah: int, output_dir: str):
    save_dir = os.path.join(output_dir, nama_kelas)
    os.makedirs(save_dir, exist_ok=True)

    per_keyword = jumlah // len(keyword_list)
    counter = 1

    for keyword in keyword_list:
        print(f"Keyword: '{keyword}' → target {per_keyword} gambar")

        google_dir = os.path.join(save_dir, f"_tmp_google_{keyword.replace(' ', '_')}")
        os.makedirs(google_dir, exist_ok=True)

        try:
            crawler = GoogleImageCrawler(
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
                storage={"root_dir": google_dir},
            )
            crawler.crawl(
                keyword=keyword,
                max_num=per_keyword,
                min_size=(100, 100),
                file_idx_offset=0,
            )
        except Exception as e:
            print(f"Google gagal: {e}")

        bing_dir = os.path.join(save_dir, f"_tmp_bing_{keyword.replace(' ', '_')}")
        os.makedirs(bing_dir, exist_ok=True)

        try:
            crawler = BingImageCrawler(
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
                storage={"root_dir": bing_dir},
            )
            crawler.crawl(
                keyword=keyword,
                max_num=per_keyword,
                min_size=(100, 100),
                file_idx_offset=0,
            )
        except Exception as e:
            print(f"Bing gagal: {e}")

        for tmp_dir in [google_dir, bing_dir]:
            for fname in sorted(os.listdir(tmp_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                src = os.path.join(tmp_dir, fname)
                dst = os.path.join(save_dir, f"{nama_kelas}_{counter:04d}{ext}")
                os.rename(src, dst)
                counter += 1

            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    print(f"Total gambar '{nama_kelas}' tersimpan: {counter - 1}")


def main():
    print("=" * 50)
    print("SCRAPING DATASET MOBIL vs MOTOR")
    print("=" * 50)

    for nama_kelas, keywords in KELAS.items():
        print(f"Kelas: {nama_kelas.upper()}")
        scrape_kelas(nama_kelas, keywords, JUMLAH_GAMBAR, OUTPUT_DIR)

    print("=" * 50)
    print("RINGKASAN DATASET")
    print("=" * 50)

    for kelas in KELAS:
        folder = os.path.join(OUTPUT_DIR, kelas)
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")]
        print(f"{kelas:10s}: {len(files)} gambar → {folder}/")

    print("Scraping selesai")


if __name__ == "__main__":
    main()