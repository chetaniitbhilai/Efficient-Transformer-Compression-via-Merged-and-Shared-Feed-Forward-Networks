
# **code_n_all — Layer Merging & Structural Compression Experiments on GPT-2**

This notebook implements and evaluates two structural modification strategies for compressing GPT-2 transformer blocks: **layer merging** and **layer deletion–based compression**.
Both methods operate on the feed-forward (FFN) layers of GPT-2, using four compression scales: **2**.

The notebook systematically evaluates these model variants across language modeling, classification, summarization, and inference-latency tasks.

---

## 🔧 **Model Variants Included**

The notebook generates and evaluates **two types of compressed models** for each of the n-values:

---

### **1. merged_n Models (2)**

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

### **2. compressed_n Models (2)**

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

* merged_2
* compressed_2
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
* merged_2
* compressed_2

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

