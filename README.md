# GSLC-Deep-Learning

# Tugas 05 – Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

**Nama:** Johanes Rhesa Sandjaja
**NIM:** 2802407875

Repositori ini berisi eksperimen perbandingan pelatihan from scratch, feature extraction,
progressive fine-tuning, uji kontrol augmentasi, dan arsitektur pembanding (EfficientNet-B0)
untuk klasifikasi 10 ras pada dataset Oxford-IIIT Pet.

## Struktur Repositori

| File | Keterangan |
|---|---|
| `GSLC_Deep_Learning.ipynb` | Notebook lengkap beserta seluruh output |
| `GSLC Deep Learning.pdf` | Laporan analisis |
| `requirements.txt` | Daftar dependensi beserta versinya |
| `metrics.csv` | Metrik akhir B1–B5 (params, LR, epoch, waktu, val/test accuracy, Macro F1) |
| `validation.csv` | Metrik validasi sebelum evaluasi test |
| `selection.csv` | Model terpilih berdasarkan validation loss |
| `architecture_comparison.csv` | Perbandingan B4_noaug (ResNet-18) dan B5 (EfficientNet-B0) |
| `B*_history.csv` | Train loss, validation loss, dan validation accuracy per epoch |
| `B*_predictions.csv` | Prediksi dan confidence tiap citra test |
| `split.csv` | Pembagian train/validation/test |
| `failure_analysis.csv` | Daftar citra test yang salah diprediksi model terbaik (B5) |
| `C5_samples.csv`, `C5_feature_maps.png` | Sampel dan visualisasi aktivasi layer4 (C5) |
| `loss.png`, `kesalahan.png` | Kurva loss dan contoh misklasifikasi |

Catatan: notebook menyimpan seluruh log ke folder `hasil/`. Pada repositori ini,
file-file tersebut diletakkan langsung di root.

## Dataset

Oxford-IIIT Pet, 10 kelas: Abyssinian, Bengal, Birman, British Shorthair, Siamese,
beagle, boxer, chihuahua, pug, samoyed (1.996 citra). Anotasi trainval dan test digabung,
lalu dibagi ulang secara stratified menjadi 1.397 train, 299 validation, dan 300 test.

## Cara Menjalankan

1. Buka notebook di Google Colab dan pilih runtime GPU (T4).
2. Install dependensi bila diperlukan: `pip install -r requirements.txt`
3. Jalankan seluruh sel (Runtime → Run all). Dataset diunduh otomatis melalui torchvision.

## Konfigurasi Eksperimen

- Seed 42 untuk Python, NumPy, dan PyTorch (cuDNN deterministic)
- Input 224 × 224 dengan normalisasi ImageNet, batch size 32
- AdamW (weight decay 0,01), CrossEntropyLoss
- Early stopping pada validation loss (patience 4, maksimal 25 epoch)
- Augmentasi train: RandomResizedCrop, horizontal flip, rotasi 15°, ColorJitter
- Hardware: Tesla T4, PyTorch 2.11.0

## Ringkasan Hasil

| Run | Strategi | Aug | Trainable Params | Val Acc | Test Acc | Test Macro F1 |
|---|---|---|---|---|---|---|
| B1 | From scratch (ResNet-18) | Ya | 11.181.642 | 51,84% | 52,00% | 0,5076 |
| B2 | Feature extraction (ResNet-18) | Ya | 5.130 | 95,65% | 97,67% | 0,9767 |
| B3 | Fine-tuning layer4 + fc (ResNet-18) | Ya | 8.398.858 | 95,99% | 98,67% | 0,9867 |
| B4_noaug | Feature extraction (ResNet-18) | Tidak | 5.130 | 95,65% | 97,33% | 0,9733 |
| B5 | Feature extraction (EfficientNet-B0) | Tidak | 12.810 | 97,99% | 98,67% | 0,9867 |

B2 sekaligus menjadi skenario B4 dengan augmentasi. B5 menggunakan konfigurasi B4_noaug
karena run tersebut memiliki validation loss terendah di antara run ResNet-18.
Pembahasan lengkap tersedia di laporan PDF.
