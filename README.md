````markdown
# H1D024046-PraktikumKB-Pertemuan6

Pengumpulan tugas praktikum Kecerdasan Buatan pertemuan 6.

# Implementasi Jaringan Syaraf Tiruan (JST)

Program ini mengimplementasikan dua algoritma Jaringan Syaraf Tiruan (JST) menggunakan bahasa pemrograman Python, yaitu:

- Perceptron untuk menyelesaikan masalah OR
- Backpropagation untuk menyelesaikan masalah XOR

---

# 1. Studi Kasus 1 — Masalah OR dengan Perceptron

(Perceptron.py & Perceptron_or.py)

Program ini mengimplementasikan model Perceptron berlayer tunggal untuk menyelesaikan masalah logika OR dengan data bipolar.

## a. Parameter Pelatihan

- Input & Target : Bipolar (-1 dan 1)
- Learning Rate (α) : 0.1
- Bobot Awal : 0
- Max Epoch : 10

## b. Data Latih (Tabel Kebenaran OR Bipolar)

| x1 | x2 | t |
|----|----|----|
| 1 | 1 | 1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

## c. Arsitektur Kelas Perceptron

| No | Fungsi | Deskripsi |
|----|---------|------------|
| 1 | `__init__()` | Konstruktor — menyimpan learning rate dan max epoch |
| 2 | `weighted_sum()` | Menghitung nilai y_in = b + Σ(xᵢ × wᵢ) |
| 3 | `predict()` | Menerapkan fungsi aktivasi bipolar |
| 4 | `plot_decision_boundary()` | Memvisualisasikan garis pemisah data setiap epoch |
| 5 | `fit()` | Fungsi utama — menjalankan proses pelatihan Perceptron |

## d. Proses Pelatihan (Fungsi fit())

1. Inisialisasi bobot dan bias = 0  
2. Untuk setiap epoch, iterasi seluruh pasang input dan target  
3. Hitung nilai:

```python
y_in = b + Σ(xᵢ × wᵢ)
```

4. Aktivasi menggunakan fungsi bipolar:

```python
y = 1 jika y_in ≥ 0
y = -1 jika y_in < 0
```

5. Hitung error:

```python
error = target - y
```

6. Jika error ≠ 0, lakukan update bobot dan bias:

```python
Δwᵢ = α × (tᵢ - yᵢ) × xᵢ
w_baru = w_lama + Δwᵢ

Δb = α × (tᵢ - yᵢ)
b_baru = b_lama + Δb
```

7. Hitung Sum Square Error (SSE)

8. Pelatihan berhenti jika:
- SSE = 0
- atau max epoch tercapai

## e. Output Program

- Grafik Decision Boundary ditampilkan menggunakan matplotlib
- Hasil pelatihan disimpan pada file:

```text
HasilPerceptron.txt
```

---

# 2. Studi Kasus 2 — Masalah XOR dengan Backpropagation

(Backpropagation.py & Backpropagation_xor.py)

Program ini mengimplementasikan model Backpropagation multi-layer untuk menyelesaikan masalah logika XOR dengan data bipolar.

Masalah XOR tidak dapat diselesaikan oleh Perceptron berlayer tunggal karena data tidak bersifat linearly separable, sehingga dibutuhkan hidden layer.

## a. Parameter Pelatihan

- Input & Target : Bipolar (-1 dan 1)
- Learning Rate (α) : 0.3
- Max Epoch : 384
- Target Error (SSE) : 0.001
- Bobot & Bias Awal : Random (0 sampai 1)

## b. Data Latih (Tabel Kebenaran XOR Bipolar)

| x1 | x2 | t |
|----|----|----|
| 1 | 1 | -1 |
| 1 | -1 | 1 |
| -1 | 1 | 1 |
| -1 | -1 | -1 |

## c. Arsitektur Jaringan

| Layer | Jumlah Neuron | Keterangan |
|--------|----------------|------------|
| Input Layer | 2 | Menerima input x1 dan x2 |
| Hidden Layer | 2 | Lapisan tersembunyi |
| Output Layer | 1 | Menghasilkan output y |

## d. Arsitektur Kelas Backpropagation

| No | Fungsi | Deskripsi |
|----|---------|------------|
| 1 | `__init__()` | Konstruktor — menyimpan parameter dan inisialisasi bobot random |
| 2 | `bi_sigmoid()` | Fungsi aktivasi sigmoid bipolar / tanh |
| 3 | `deriv_bi_sigmoid()` | Turunan fungsi tanh |
| 4 | `plot_error()` | Memvisualisasikan penurunan SSE setiap epoch |
| 5 | `fit()` | Fungsi utama — menjalankan forward dan backward propagation |

## e. Proses Pelatihan (Fungsi fit())

### Forward Propagation

1. Hitung hidden input:

```python
h_in = b_hidden + Σ(xᵢ × w_hiddenᵢ)
```

2. Aktivasi hidden layer:

```python
h = tanh(h_in)
```

3. Hitung output input:

```python
y_in = b_output + Σ(hᵢ × w_outputᵢ)
```

4. Aktivasi output layer:

```python
y = tanh(y_in)
```

### Backward Propagation

5. Hitung error output:

```python
error = target - y
```

6. Hitung delta output:

```python
δ_output = error × (1 - y²)
```

7. Hitung error hidden:

```python
error_hidden = Σ(δ_output × w_outputᵀ)
```

8. Hitung delta hidden:

```python
δ_hidden = error_hidden × (1 - h²)
```

9. Perbarui bobot dan bias output layer:

```python
Δw_output = Σ(hᵀ × δ_output) × α
Δb_output = Σ(δ_output) × α
```

10. Perbarui bobot dan bias hidden layer:

```python
Δw_hidden = Σ(x × δ_hidden) × α
Δb_hidden = Σ(δ_hidden) × α
```

11. Hitung SSE rata-rata

12. Pelatihan berhenti jika:
- SSE < target error
- atau max epoch tercapai

## f. Output Program

- Grafik penurunan SSE ditampilkan menggunakan matplotlib
- Hasil pelatihan disimpan pada file:

```text
hasilBackpropagation.txt
```

---

# 3. Fungsi Aktivasi yang Digunakan

| Fungsi Aktivasi | Rumus | Digunakan Pada |
|-----------------|--------|----------------|
| Bipolar (Step) | y = 1 jika y_in ≥ 0, y = -1 jika y_in < 0 | Perceptron |
| Sigmoid Bipolar (tanh) | y = tanh(y_in) | Backpropagation |
| Turunan tanh | y' = 1 - y² | Backpropagation |

---

# 4. Library yang Digunakan

## a. numpy

Digunakan untuk operasi matriks dan array numerik, seperti:

- `np.dot()`
- `np.random.rand()`
- operasi vektor dan matriks lainnya

## b. matplotlib

Digunakan untuk visualisasi grafik:

- Decision Boundary pada Perceptron
- Grafik penurunan SSE pada Backpropagation

---

# 5. Struktur File

| File | Deskripsi |
|------|------------|
| `Perceptron.py` | Definisi kelas Perceptron |
| `Perceptron_or.py` | File eksekusi Perceptron |
| `Backpropagation.py` | Definisi kelas Backpropagation |
| `Backpropagation_xor.py` | File eksekusi Backpropagation |
| `HasilPerceptron.txt` | Output hasil pelatihan Perceptron |
| `hasilBackpropagation.txt` | Output hasil pelatihan Backpropagation |

---

# 6. Cara Menjalankan Program

## Install dependencies

```bash
pip install numpy matplotlib
```

## Jalankan Studi Kasus 1 — Perceptron

```bash
python Perceptron_or.py
```

## Jalankan Studi Kasus 2 — Backpropagation

```bash
python Backpropagation_xor.py
```
````
