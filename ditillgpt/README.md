
# **code_distill — DistilGPT-2 Evaluation Pipeline**

This notebook evaluates the distilled version of GPT-2 (**DistilGPT-2**) across language modeling, classification, and summarization tasks.
Unlike pruning or merging-based compression, distillation uses a *pre-trained, reduced* variant of GPT-2 that is already smaller, faster, and lighter. This notebook focuses on analyzing its performance relative to the standard GPT-2 family.

---

## 🔧 **Model Variant Included**

### **1. DistilGPT-2**

This model is a **compressed and distilled** version of GPT-2, trained using knowledge distillation.
It preserves much of GPT-2’s behavior while reducing size and improving speed.

#### Characteristics:

* Fewer transformer blocks
* Fewer parameters than GPT-2
* Faster inference
* More efficient fine-tuning
* Maintains good perplexity and downstream accuracy

#### Purpose:

* Benchmark distillation vs. pruning/merging
* Examine trade-offs between model size and performance
* Provide a lightweight LM baseline for downstream tasks

---

## 📘 **Notebook Workflow**

### **1. Load DistilGPT-2**

Loads the model and tokenizer from HuggingFace and prints:

* total parameters
* trainable parameters

This helps quantify the size reduction achieved by distillation.

---

### **2. Perplexity Evaluation**

Computes perplexity on a language-modeling dataset using:

* evaluation loop
* cross-entropy loss
* exponential transformation for perplexity

This evaluates the core LM quality of DistilGPT-2.

---

### **3. Classification Evaluation**

Fine-tunes DistilGPT-2 on a classification task (e.g., IMDb sentiment) and reports:

* evaluation accuracy
* classification performance relative to GPT-2 variants

This measures discriminative downstream capability.

---

### **4. Summarization (ROUGE-L Evaluation)**

Evaluates the model on a summarization dataset and prints:

* ROUGE-L score for DistilGPT-2

This provides a generative downstream comparison against other compressed models.

---

## 🎯 **Purpose of the Notebook**

This notebook serves as a **baseline distillation benchmark** for your compression experiments.
It enables comparison of DistilGPT-2 against:

* merged_n models (4,6,8,10)
* compressed_n models (4,6,8,10)
* pruned_104M model
* original GPT-2

The notebook provides a unified evaluation across:

* perplexity
* classification accuracy
* summarization performance

allowing direct analysis of how **distillation compares to structural and sparsity-based compression techniques**.

---

If you'd like, I can produce unified README headers, diagrams, or a combined project-level README for all three notebooks.
