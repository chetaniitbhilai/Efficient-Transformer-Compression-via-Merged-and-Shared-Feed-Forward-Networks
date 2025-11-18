
# **Pruned_104M — Weight Pruning Based Compression Pipeline**

This notebook implements and evaluates **weight pruning** on GPT-2 to create a lighter, sparsified model with approximately **104M effective parameters**.
Unlike layer merging or structural compression, pruning keeps the architecture **unchanged** but removes a large number of individual weights, increasing sparsity while preserving model depth.

The notebook analyzes the pruned model’s performance across language modeling, classification, and summarization tasks.

---

## 🔧 **Model Variant Included**

### **1. pruned_104M**

This model is created by applying **global unstructured weight pruning** to GPT-2.

#### What happens:

* Identify the lowest-magnitude weights across Linear layers
* Zero out (remove) those weights while keeping the tensor shapes intact
* Maintain the same number of layers, hidden dimensions, and depth
* Preserve GPT-2 architecture for compatibility
* Significantly reduce the number of active parameters

#### Purpose:

* Achieve parameter reduction while keeping the model structure intact
* Introduce sparsity-based compression
* Compare pruning effectiveness with merging, deletion-compression, and distillation

---

## 📘 **Notebook Workflow**

### **1. Load GPT-2**

Initializes the original GPT-2 checkpoint and prepares it for pruning and evaluation.

---

### **2. Perplexity Evaluation (Before Pruning)**

Computes the baseline perplexity of the full model using:

* cross-entropy on a validation dataset
* exponential conversion to perplexity

This serves as a reference before any compression is applied.

---

### **3. Apply Weight Pruning**

Performs global magnitude pruning across relevant GPT-2 layers:

* Generates pruning masks
* Applies masks to zero out low-magnitude weights
* Computes the count of remaining non-zero parameters
* Produces a sparse GPT-2 variant with ~104M active parameters

---

### **4. Perplexity Evaluation (After Pruning)**

Re-evaluates perplexity with the pruned model to measure:

* immediate language modeling degradation
* sensitivity of GPT-2 to sparsity

---

### **5. Classification Evaluation**

Tests the pruned model on a classification task (e.g., IMDb):

* attaches a classification head
* computes evaluation accuracy
* compares behavior against structurally compressed and distilled models

---

### **6. Summarization (ROUGE-L Evaluation)**

Evaluates the pruned model on a summarization dataset and prints:

* ROUGE-L score

This checks the model’s generative downstream capability under heavy sparsity.

---

## 🎯 **Purpose of the Notebook**

This notebook provides a complete experimental setup for evaluating **sparsity-based compression** in GPT-2.
It allows comparison of pruned_104M against:

* merged_n models (4, 6, 8, 10)
* compressed_n models (4, 6, 8, 10)
* distilGPT-2
* the original GPT-2 baseline

The included evaluations—perplexity, classification, ROUGE-L, and parameter counting—enable a full analysis of how pruning affects:

* language modeling quality
* downstream task performance
* compression–accuracy tradeoffs

