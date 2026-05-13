# H1D024046-PraktikumKB-Pertemuan6
Repository ini berisi hasil implementasi tugas Praktikum Kecerdasan Buatan Pertemuan 6 mengenai penerapan dasar Jaringan Syaraf Tiruan (JST) menggunakan bahasa pemrograman Python.

## Deskripsi Program
Pada praktikum ini dilakukan implementasi dua metode JST, yaitu:

1. **Perceptron** untuk menyelesaikan kasus logika OR.
2. **Backpropagation** untuk menyelesaikan kasus logika XOR.

Kedua program dibuat menggunakan library `numpy` untuk proses komputasi numerik dan `matplotlib` untuk visualisasi hasil pelatihan.

---

# 1. Implementasi Perceptron pada Logika OR
File yang digunakan:
- `Perceptron.py`
- `Perceptron_or.py`

## Penjelasan
Program ini menggunakan metode Single Layer Perceptron untuk mempelajari pola logika OR bipolar. Model akan melakukan proses pelatihan dengan memperbaiki bobot dan bias hingga error bernilai minimum atau epoch mencapai batas maksimum.

## Parameter Pelatihan
| Parameter | Nilai |
|---|---|
| Learning Rate | 0.1 |
| Maksimum Epoch | 10 |
| Input | Bipolar (-1 dan 1) |
| Bobot Awal | 0 |

## Dataset OR Bipolar
| x1 | x2 | Target |
|---|---|---|
| 1 | 1 | 1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

## Fungsi pada Class Perceptron
| Fungsi | Keterangan |
|---|---|
| `__init__()` | Menyimpan parameter learning rate dan epoch |
| `weighted_sum()` | Menghitung nilai net input |
| `predict()` | Menentukan output menggunakan aktivasi bipolar |
| `plot_decision_boundary()` | Menampilkan garis pemisah data |
| `fit()` | Menjalankan proses training Perceptron |

## Alur Pelatihan
1. Inisialisasi bobot dan bias dengan nilai awal 0.
2. Hitung nilai input bersih (`y_in`).
3. Terapkan fungsi aktivasi bipolar.
4. Hitung error antara target dan output.
5. Perbarui bobot dan bias menggunakan aturan pembelajaran Perceptron.
6. Hitung nilai SSE setiap epoch.
7. Proses berhenti jika SSE = 0 atau epoch maksimum tercapai.

## Output Program
- Visualisasi decision boundary setiap epoch.
- File hasil perhitungan bernama `HasilPerceptron.txt`.

## Hasil Visualisasi
### Grafik Decision Boundary
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/4e2e6708-22b7-4d55-865b-b68866f19d8f" />

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/47e75565-831c-4abf-b955-6c13a67cabdf" />

<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/86ea3606-a9a3-48df-9e37-5a12deaf6caf" />

---

# 2. Implementasi Backpropagation pada Logika XOR
File yang digunakan:
- `Backpropagation.py`
- `Backpropagation_xor.py`

## Penjelasan
Program ini menerapkan metode Backpropagation untuk menyelesaikan logika XOR bipolar. Karena XOR tidak dapat dipisahkan secara linear, maka digunakan hidden layer agar model mampu mempelajari pola data.

## Parameter Pelatihan
| Parameter | Nilai |
|---|---|
| Learning Rate | 0.3 |
| Maksimum Epoch | 1000 |
| Target Error | 0.001 |
| Bobot Awal | Random |

## Dataset XOR Bipolar
| x1 | x2 | Target |
|---|---|---|
| 1 | 1 | -1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

## Struktur Jaringan
| Layer | Jumlah Neuron |
|---|---|
| Input Layer | 2 |
| Hidden Layer | 2 |
| Output Layer | 1 |

## Fungsi pada Class Backpropagation
| Fungsi | Keterangan |
|---|---|
| `__init__()` | Inisialisasi parameter dan bobot |
| `bi_sigmoid()` | Fungsi aktivasi tanh |
| `deriv_bi_sigmoid()` | Turunan fungsi tanh |
| `plot_error()` | Menampilkan grafik penurunan error |
| `fit()` | Menjalankan proses training Backpropagation |

## Tahapan Proses Training
### Forward Propagation
1. Input dihitung menuju hidden layer.
2. Hidden layer diaktivasi menggunakan fungsi tanh.
3. Nilai hidden diteruskan ke output layer.
4. Output akhir diaktivasi menggunakan tanh.

### Backward Propagation
1. Menghitung error output.
2. Menghitung delta output.
3. Menghitung error hidden layer.
4. Mengupdate bobot dan bias.
5. Menghitung SSE pada setiap epoch.

## Output Program
- Grafik penurunan error SSE.
- File hasil pelatihan bernama `hasilBackpropagation.txt`.

## Hasil Visualisasi
### Grafik Penurunan Error
<img width="640" height="480" alt="Figure_4" src="https://github.com/user-attachments/assets/c1a6802d-43c9-4ce5-87e2-30c443d4b4f1" />

---

# Fungsi Aktivasi yang Digunakan
| Fungsi Aktivasi | Rumus |
|---|---|
| Step Bipolar | y = 1 jika y_in ≥ 0, dan y = -1 jika y_in < 0 |
| Sigmoid Bipolar (tanh) | y = tanh(x) |
| Turunan tanh | y' = 1 - y² |

---

# Library yang Digunakan
| Library | Fungsi |
|---|---|
| `numpy` | Operasi matriks dan perhitungan numerik |
| `matplotlib` | Visualisasi grafik hasil pelatihan |

---

# Struktur File
| File | Keterangan |
|---|---|
| `Perceptron.py` | Class Perceptron |
| `Perceptron_or.py` | Program utama kasus OR |
| `Backpropagation.py` | Class Backpropagation |
| `Backpropagation_xor.py` | Program utama kasus XOR |
| `HasilPerceptron.txt` | Output hasil training Perceptron |
| `hasilBackpropagation.txt` | Output hasil training Backpropagation |

---

# Cara Menjalankan Program

## Install Library
```bash
pip install numpy matplotlib
```

## Menjalankan Perceptron
```bash
python Perceptron_or.py
```

## Menjalankan Backpropagation
```bash
python Backpropagation_xor.py
```
