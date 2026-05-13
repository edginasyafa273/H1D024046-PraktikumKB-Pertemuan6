# H1D024046-PraktikumKB-Pertemuan6
Repositori pengumpulan tugas praktikum mata kuliah Kecerdasan Buatan (Pertemuan 6).

## Implementasi Jaringan Syaraf Tiruan (JST) Dasar
Proyek ini memuat kode implementasi dua buah model komputasi Jaringan Syaraf Tiruan (JST) yang ditulis menggunakan Python. Model yang dibuat mencakup algoritma Perceptron guna memecahkan logika OR, serta algoritma Backpropagation untuk menangani logika XOR.

### 1. Eksperimen 1: Logika OR dengan Model Perceptron (`Perceptron.py` & `Perceptron_or.py`)
Pada bagian ini, dibangun sebuah model JST dengan arsitektur *Single-Layer Perceptron* yang bertujuan mengklasifikasikan operasi logika OR. Data yang diproses direpresentasikan dalam bentuk bipolar.

#### a. Konfigurasi Pelatihan
- **Nilai Input & Target** : Bipolar (-1 dan 1)
- **Tingkat Pembelajaran (Learning Rate / α)** : 0.1
- **Inisialisasi Bobot** : 0
- **Batas Maksimal Epoch** : 10

#### b. Dataset Pelatihan (Tabel Logika OR Bipolar)
| Input 1 (x1) | Input 2 (x2) | Target (t) |
|---|---|---|
| 1 | 1 | 1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

#### c. Rancang Bangun Class Perceptron
| No | Nama Fungsi | Keterangan |
|---|---|---|
| 1 | `__init__()` | Method konstruktor untuk menyimpan pengaturan *learning rate* serta batas *epoch* |
| 2 | `weighted_sum()` | Menghitung akumulasi nilai masuk: `y_in = b + Σ(x_i × w_i)` |
| 3 | `predict()` | Menjalankan *step function* bipolar untuk menentukan hasil akhir (1 atau -1) |
| 4 | `plot_decision_boundary()` | Menampilkan plot visual untuk garis batas keputusan tiap epoch |
| 5 | `fit()` | Fungsi inti penentu jalannya proses *training* model Perceptron |

#### d. Mekanisme Training (Fungsi `fit()`)
1. Mulai dengan memberikan nilai 0 pada bobot maupun bias.
2. Di setiap perulangan (*epoch*), lakukan iterasi pada tiap baris pasangan input dan target.
3. Lakukan kalkulasi `y_in = b + Σ(x_i × w_i)`.
4. Masukkan nilai `y_in` ke fungsi aktivasi bipolar guna mendapatkan nilai keluaran (`y`).
5. Hitung selisih atau *error* dengan rumus: `error = target - y`.
6. Apabila ditemukan `error ≠ 0`, sesuaikan bobot dan bias menggunakan metode *Delta Rule*:
   - Penyesuaian bobot: `Δw_i = α × (t_i - y_i) × x_i`
   - Pembaruan bobot: `w_i(baru) = w_i(lama) + Δw_i`
   - Penyesuaian bias: `Δb = α × (t_i - y_i)`
   - Pembaruan bias: `b(baru) = b(lama) + Δb`
7. Kalkulasi nilai *Sum Square Error* (SSE). Hentikan pelatihan jika target SSE mencapai 0 atau sudah menyentuh batas maksimum *epoch*.

#### e. Hasil Eksekusi Program
- Memunculkan grafik *Decision Boundary* secara visual (memanfaatkan `matplotlib`) pada setiap epoch.
- Rincian hasil komputasi yang meliputi nilai bobot, bias, serta besaran error di tiap langkah iterasi akan disimpan ke dalam berkas `HasilPerceptron.txt`.

#### f. Cuplikan Grafik Perceptron
![Grafik Perceptron](outputgrafik1.jpeg)

---

### 2. Eksperimen 2: Logika XOR dengan Model Backpropagation (`Backpropagation.py` & `Backpropagation_xor.py`)
Kode ini menyajikan model JST jenis Backpropagation yang memiliki arsitektur *multi-layer* guna menyelesaikan kendala pemisahan non-linear pada gerbang XOR. Karena kasus XOR tergolong data yang tidak *linearly separable*, model butuh disisipi oleh *hidden layer*.

#### a. Konfigurasi Pelatihan
- **Nilai Input & Target** : Bipolar (-1 dan 1)
- **Tingkat Pembelajaran (Learning Rate / α)** : 0.3
- **Batas Maksimal Epoch** : 1000
- **Batas Toleransi Error (SSE)** : 0.001
- **Inisialisasi Bobot & Bias** : Acak (antara rentang 0 hingga 1)

#### b. Dataset Pelatihan (Tabel Logika XOR Bipolar)
| Input 1 (x1) | Input 2 (x2) | Target (t) |
|---|---|---|
| 1 | 1 | -1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

#### c. Arsitektur Jaringan Multi-Layer
| Lapisan (Layer) | Total Neuron | Peran |
|---|---|---|
| Input Layer | 2 node | Bertugas menerima masukan `x1` dan `x2` |
| Hidden Layer | 2 node | Menjadi lapisan penyembunyi perantara komputasi |
| Output Layer | 1 node | Merumuskan keluaran atau hasil prediksi (`y`) |

#### d. Rancang Bangun Class Backpropagation
| No | Nama Fungsi | Keterangan |
|---|---|---|
| 1 | `__init__()` | Konstruktor untuk mengatur parameter dasar dan inisiasi bobot *random* |
| 2 | `bi_sigmoid()` | Bertindak sebagai fungsi aktivasi sigmoid jenis bipolar (fungsi *tanh*) |
| 3 | `deriv_bi_sigmoid()` | Merupakan fungsi turunan dari *tanh* yang dipakai saat propagasi balik |
| 4 | `plot_error()` | Membuat plot grafik tren penurunan SSE selama iterasi epoch |
| 5 | `fit()` | Menjadi fungsi sentral pengeksekusi urutan *forward* dan *backward propagation* |

#### e. Mekanisme Training (Fungsi `fit()`)
**Tahap Propagasi Maju (Forward Propagation):**
1. Lakukan operasi antara input layer dan hidden layer: `h_in = b_hidden + Σ(x_i × w_hidden_i)`
2. Terapkan fungsi aktivasi *tanh* pada hasil lapisan tersembunyi: `h = tanh(h_in)`
3. Lakukan operasi antara hidden layer dan output layer: `y_in = b_output + Σ(h_i × w_output_i)`
4. Terapkan fungsi aktivasi *tanh* untuk nilai akhir: `y = tanh(y_in)`

**Tahap Propagasi Balik (Backward Propagation):**
5. Evaluasi besar error di titik output: `error = target - y`
6. Kalkulasi faktor delta pada output: `δ_output = error × (1 - y²)`
7. Evaluasi besaran error yang terjadi pada hidden layer: `error_hidden = Σ(δ_output_i × w_output_i^T)`
8. Kalkulasi faktor delta pada hidden layer: `δ_hidden = error_hidden × (1 - h²)`
9. Update matriks bobot serta bias untuk lapisan output:
   - `Δw_output = Σ(h_i^T × δ_output_i) × α`
   - `Δb_output = Σ(δ_output_i) × α`
10. Update matriks bobot serta bias untuk lapisan tersembunyi:
   - `Δw_hidden = Σ(x_i × δ_hidden_i) × α`
   - `Δb_hidden = Σ(δ_hidden_i) × α`
11. Kalkulasi rerata *Sum Square Error* (SSE). Lakukan stop iterasi bila nilai SSE telah di bawah batas minimum (*target error*) atau jika putaran telah menyentuh *max epoch*.

#### f. Hasil Eksekusi Program
- Menampilkan grafik visual penurunan nilai error SSE per iterasi menggunakan `matplotlib`.
- Rincian komputasi maju-mundur secara keseluruhan (termasuk update bobot maupun rekap nilai SSE) dicetak dan diamankan ke dalam `hasilBackpropagation.txt`.

#### g. Cuplikan Grafik Backpropagation
![Grafik Backpropagation](outputgrafik2.jpeg)

---

### 3. Ragam Fungsi Aktivasi
| Nama Fungsi Aktivasi | Formula Matematika | Implementasi Penggunaan |
|---|---|---|
| Step Bipolar | `y = 1` bila `y_in ≥ 0`, `y = -1` bila `y_in < 0` | Metode Perceptron (`predict()`) |
| Tanh (Sigmoid Bipolar) | `y = tanh(y_in)` | Metode Backpropagation (`bi_sigmoid()`) |
| Turunan fungsi Tanh | `y' = 1 - y²` | Metode Backpropagation (`deriv_bi_sigmoid()`) |

### 4. Pustaka (Library) Tambahan
- **`numpy`**: Berperan sentral dalam menangani kalkulasi matriks, array, memfasilitasi perkalian *dot product* lewat `np.dot()`, menghasilkan pembobotan acak via `np.random.rand()`, serta beberapa manajemen vektor lainnya.
- **`matplotlib`**: Berperan di sektor visual data guna menyajikan:
  - Pola garis pemisah (*Decision Boundary*) setiap proses iterasi dalam Perceptron.
  - Kurva penurunan tingkat kesalahan (*Sum Square Error*) sepanjang masa *training* di Backpropagation.

### 5. Susunan Direktori File
| Nama Berkas | Penjelasan Singkat |
|---|---|
| `Perceptron.py` | Tempat di mana arsitektur Class Perceptron didefinisikan |
| `Perceptron_or.py` | *Script runner* guna menyetel data kasus OR lalu memicu pelatihan model Perceptron |
| `Backpropagation.py` | Tempat di mana arsitektur Class Backpropagation diatur sedemikian rupa |
| `Backpropagation_xor.py` | *Script runner* yang mendefinisikan dataset XOR kemudian melatih jaringan Backpropagation |
| `outputgrafik1.jpeg` | Contoh hasil *render* grafik Decision Boundary Perceptron |
| `outputgrafik2.jpeg` | Contoh hasil *render* visualisasi tren penurunan error pada model Backpropagation |
| `HasilPerceptron.txt` | *Log file* pencatatan kalkulasi Perceptron (auto-generated saat *script* dijalankan) |
| `hasilBackpropagation.txt` | *Log file* riwayat perhitungan Backward/Forward propagation (auto-generated saat dieksekusi) |

### 6. Panduan Menjalankan Kode
```bash
# Lakukan instalasi modul yang wajib ada
pip install numpy matplotlib

# Eksekusi Eksperimen 1: Model Perceptron (Logika OR)
python Perceptron_or.py

# Eksekusi Eksperimen 2: Model Backpropagation (Logika XOR)
python Backpropagation_xor.py
```
