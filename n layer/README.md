

# Soft Shared FFN with Low-Rank Deltas for GPT-2

This repository implements a **soft parameter sharing method for GPT-2 feed-forward networks (FFNs)** using a **shared base weight plus low-rank, layer-specific residuals**.
The approach reduces redundancy across neighboring transformer layers **without averaging weights, using MoE routing, or discarding information**.

---

## Core Idea

Transformer FFNs across nearby layers often learn **highly correlated neuron subspaces**.
Instead of merging or pruning them, we:

1. **Align FFN neurons** across adjacent layers using activation correlation and Hungarian matching.
2. **Share a common base FFN weight** (`Wc`) across a sliding window of layers.
3. **Add a learnable low-rank delta** (`A @ B`) per layer to preserve expressivity.
4. **Fine-tune only the low-rank residuals** with a regularized objective.

Mathematically, each FFN linear layer becomes:

[
W_l = W_c + A_l B_l \quad\text{with}\quad \text{rank}(A_l B_l) \ll d
]

---

## Method Overview

**Pipeline**

1. Extract FFN pre-activations (`c_fc`) from GPT-2 in a single forward pass.
2. Compute neuron alignment using **correlation + Hungarian assignment**.
3. Search for the best contiguous layer window via **perplexity evaluation**.
4. Replace FFNs with **shared base + low-rank deltas**.
5. Fine-tune using:

   * Language modeling loss
   * Frobenius-norm regularization on low-rank updates

---

## Key Properties

* ✅ No hard parameter tying
* ✅ No mixture-of-experts routing
* ✅ No weight averaging
* ✅ Fully differentiable
* ✅ Drop-in replacement for GPT-2 FFNs

---

## Configuration

```python
RANK = 8          # Low-rank dimension
WINDOW_K = 4     # Number of adjacent layers to share
LAMBDA = 1e-4    # Low-rank regularization
MAX_TOKENS = 4000
```

---

## Results (Qualitative)

* Perplexity remains **close to or better than baseline** after recovery.
* Model retains generation quality after fine-tuning.
* Demonstrates **redundancy-aware parameter sharing** in transformer FFNs.

---

## Use Cases

* Model compression research
* Studying neuron alignment across layers
* Soft parameter sharing in transformers
* Low-rank adaptation beyond LoRA-style adapters

---

## Notes

* Designed for **GPT-2**, but extensible to other decoder-only transformers.
* Alignment is local (sliding window) to avoid global instability.
* Low-rank deltas ensure **no expressivity collapse**.

---

## Example Generation

```text
India will become global leader in AI because …
```

---

## Status

**Research prototype** – intended for experimentation, analysis, and extensions (e.g., LLaMA, ViT, TinyViT).







# **code_n_all — Layer Merging & Structural Compression Experiments on GPT-2**

This notebook implements and evaluates two structural modification strategies for compressing GPT-2 transformer blocks: **layer merging** and **layer deletion–based compression**.
Both methods operate on the feed-forward (FFN) layers of GPT-2, using four compression scales: **4, 6, 8, 10**.

The notebook systematically evaluates these model variants across language modeling, classification, summarization, and inference-latency tasks.

---

## 🔧 **Model Variants Included**

The notebook generates and evaluates **two types of compressed models** for each of the n-values:

---

### **1. merged_n Models (4, 6, 8, 10)**

These models **average the weights** of n consecutive FFN layers but **do not remove any layers**.

#### What happens:

* Select n FFN layers
* Compute their **element-wise average**
* Replace *each* of the n layers with this averaged layer
* Total number of layers remains unchanged
* Parameter count remains the same
* Only internal structure changes

#### Purpose:

* Smooth representation
* Study how uniformizing layers affects performance
* No speed or parameter reduction expected

---

### **2. compressed_n Models (4, 6, 8, 10)**

These models **delete n–1 layers** and replace the whole group with **one averaged layer**, achieving actual compression.

#### What happens:

* Select n FFN layers
* Average their weights into a single FFN layer
* **Delete the remaining (n–1) layers**
* Insert the averaged layer in place of the block
* Model depth decreases
* Parameter count decreases
* Inference becomes faster

#### Purpose:

* Structural model compression
* Reduce model size & latency
* Evaluate quality degradation vs. speedup

---

## 📘 **Notebook Workflow**

### **1. Text Generation**

Generates sample text from:

* merged_4,6,8,10
* compressed_4,6,8,10
* original GPT-2

This provides a qualitative first look at how structural modifications affect model behavior.

---

### **2. Perplexity (Without Fine-Tuning)**

Evaluates all model variants directly after merging/compression to measure immediate language modeling shifts.

---

### **3. Fine-Tuning**

All merged and compressed models are fine-tuned using the same training loop to ensure a fair comparison.

---

### **4. Perplexity (After Fine-Tuning)**

Recalculates perplexity for:

* original
* merged_4,6,8,10
* compressed_4,6,8,10

This measures how much performance each model regains post-training.

---

### **5. Unified Perplexity Summary**

A combined evaluation loop reports perplexity for all 5 model variants (original + 4 modified).

---

### **6. Latency Measurement**

Measures average inference time for:

* merged models
* compressed models
* original model

This quantifies the practical speedup achieved by compression.

---

### **7. Latency Comparison Table**

Computes relative speedup for compressed_n models vs. the original GPT-2.

---

### **8. IMDb Sentiment Classification**

Attaches a classification head and evaluates all models on IMDb to observe discriminative downstream impact.

---

### **9. Summarization (ROUGE-L Evaluation)**

Evaluates the models on a summarization dataset to compare generative quality.

---

## 🎯 **Purpose of the Notebook**

This notebook enables systematic comparison of two structural compression techniques:

* **Weight averaging without depth reduction** (merged_n)
* **Depth-reducing structural compression** (compressed_n)

It helps in understanding:

* How merging affects representation smoothness
* How structural compression changes accuracy, perplexity, and latency
* Tradeoffs between model size, speed, and performance
* Behavior across both generative and discriminative tasks

