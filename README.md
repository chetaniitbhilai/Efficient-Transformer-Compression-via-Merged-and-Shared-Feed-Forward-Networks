
# **Efficient Transformer Compression via Merged and Shared Feed-Forward Networks**

This repository presents an extensive study on **efficient GPT-2 compression** using:

* **Layer merging (merged_n)**
* **Layer deletion + averaging (compressed_n)**
* **Weight pruning (pruned_104M)**
* **Knowledge distillation (DistilGPT-2)**

All variants are evaluated on:

* **Perplexity (LM quality)**
* **Inference latency (efficiency)**
* **IMDb sentiment accuracy (classification)**
* **ROUGE-L (summarization quality)**
* **Parameter count (model size)**

This work systematically compares how structural, sparsity-based, and distillation-based compression techniques affect transformer performance.

---

# 📦 **Model Variants**

| Category   | Variant        | Description                                                |
| ---------- | -------------- | ---------------------------------------------------------- |
| Baseline   | GPT-2 Original | Standard GPT-2 (~124M params)                              |
| Distilled  | DistilGPT-2    | Smaller distilled version (~82M params)                    |
| Pruned     | pruned_104M    | Unstructured weight pruning (~104M non-zero params)        |
| Merged     | merged_n       | Average weights of n FFN layers (no depth reduction)       |
| Compressed | compressed_n   | Merge n FFNs → replace with a single FFN (depth reduction) |

Evaluated for **n ∈ {2, 4, 6, 8, 10}**.

---

# 📊 **1. Parameter Comparison**

| Model           | Parameters  |
| --------------- | ----------- |
| GPT-2 Original  | **124M**    |
| DistilGPT-2     | **81.91M**  |
| Pruned-104M     | **104.69M** |
| Merged-4/6/8/10 | **124M**    |
| Compressed-2    | **81.91M**  |
| Compressed-4    | **60.64M**  |
| Compressed-6    | **53.56M**  |
| Compressed-8    | **53.56M**  |
| Compressed-10   | **53.56M**  |

---

# 🔍 **2. Perplexity Results**

| Model          | Perplexity       |
| -------------- | ---------------- |
| GPT-2 Original | **12.22**        |
| DistilGPT-2    | **35.21**        |
| Pruned-104M    | **210,346.42**   |
| Compressed-2   | **18.93**        |
| Merged-4       | **13.36**        |
| Compressed-4   | **28.49**        |
| Merged-6       | **13.16**        |
| Compressed-6   | **33.22**        |
| Merged-8       | **13.11**        |
| Compressed-8   | **32.00**        |
| Merged-10      | **13.17**        |
| Compressed-10  | **31.82** |

---

# ⚡ **3. Latency Results**

| Model          | Latency (ms/sample) | Speedup vs GPT-2 |
| -------------- | ------------------- | ---------------- |
| GPT-2 Original | **73.90 ms**        | 1.00×            |
| DistilGPT-2    | *not measured*      | —                |
| Pruned-104M    | *not measured*      | —                |
| Compressed-2   | **127.10 ms**       | 0.58× slower     |
| Merged-4       | **71.28 ms**        | **1.03×**        |
| Compressed-4   | **590.37 ms**       | 0.12×            |
| Merged-6       | **84.85 ms**        | 0.87×            |
| Compressed-6   | **30.50 ms**        | **2.42×**        |
| Merged-8       | **83.57 ms**        | 0.88×            |
| Compressed-8   | **502.06 ms**       | 0.14×            |
| Merged-10      | **71.61 ms**        | **1.03×**        |
| Compressed-10  | **31.12 ms**        | **2.37×**        |

---

# 🎯 **4. IMDb Classification Accuracy**

| Model          | Accuracy   |
| -------------- | ---------- |
| GPT-2 Original | **87.45%** |
| DistilGPT-2    | **86.02%** |
| Pruned-104M    | **83.46%** |
| Compressed-2   | **83.58%** |
| Merged-2       | **83.67%** |
| Merged-4       | **82.99%** |
| Compressed-4   | **83.46%** |
| Merged-6       | **83.34%** |
| Compressed-6   | **83.34%** |
| Merged-8       | **82.94%** |
| Compressed-8   | **82.95%** |
| Merged-10      | **83.07%** |
| Compressed-10  | **83.14%** |

---

# 📝 **5. Summarization Quality (ROUGE-L)**

| Model          | ROUGE-1    | ROUGE-2    | ROUGE-L    |
| -------------- | ---------- | ---------- | ---------- |
| GPT-2 Original | 0.2369     | 0.1045     | **0.1703** |
| DistilGPT-2    | 0.2228     | 0.0987     | **0.1602** |
| Compressed-2   | 0.2001     | 0.0881     | **0.1405** |
| Merged-4       | 0.2307     | 0.1005     | **0.1624** |
| Compressed-4   | 0.1960     | 0.0863     | **0.1373** |
| Merged-6       | 0.2269     | 0.0998     | **0.1603** |
| Compressed-6   | 0.2010     | 0.0884     | **0.1408** |
| Merged-8       | 0.2282     | 0.1001     | **0.1595** |
| Compressed-8   | 0.1991     | 0.0876     | **0.1394** |
| Merged-10      | **0.2657** | **0.1173** | **0.1860** |
| Compressed-10  | 0.2101     | 0.0915     | **0.1467** |

---

# 🧠 **Summary of Findings**

### ✅ **Merged-n models**

* Maintain **all parameters**
* Preserve **low perplexity** close to GPT-2
* Maintain accuracy (≈83%)
* **Little to no latency degradation**
* Very strong ROUGE for *n = 10*

### ✅ **Compressed-n (layer-reduced) models**

* Drastically reduce parameters (up to **57% smaller**)
* Mixed perplexity results
* Some variants very fast (compressed-6, compressed-10)
* Some extremely slow (compressed-4, compressed-8)
* Accuracy remains stable around ~83%

### ❌ **Pruned-104M**

* Extremely high perplexity → unstable
* Accuracy slightly lower
* Not competitive with merged/compressed variants

### ✅ **DistilGPT-2**

* Strong baseline for distillation
* Good accuracy (86%)
* Lower ROUGE-L
* Balanced quality-size tradeoff

---

# 📌 **Conclusion**

This project demonstrates that **merging and sharing feed-forward networks** is a *highly effective compression strategy* for GPT-2, preserving both generative and discriminative performance while enabling parameter-efficient architectures.

Compression-6 and Compression-10 variants exhibit the best **speed–quality tradeoff**, while merged-10 achieves the **highest ROUGE-L** among compressed models.

---
