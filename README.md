# Gaussian Naive Bayes from Scratch

> **Implementing a probabilistic classifier from first principles — no black boxes, just Bayes, Gaussians, and NumPy.**

A complete, from-scratch implementation of **Gaussian Naive Bayes (GNB)** in Python, built for the *Theoretical Foundations of Machine Learning* course at **Cairo University, Faculty of Computing and Artificial Intelligence (FCAI)**.

We do not call a ready-made classifier to learn. We derive it, code it, test it by hand, break its assumptions on purpose — and then prove it matches scikit-learn exactly.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-datasets%20%26%20comparison-orange)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.8%25-brightgreen)](GNB_20242023_20240523.ipynb)
[![sklearn agreement](https://img.shields.io/badge/sklearn%20agreement-100%25-brightgreen)](GNB_20242023_20240523.ipynb)

---

## 📊 Results at a Glance

| Metric | Value |
|---|---|
| **Test accuracy** | **0.938** — 106 / 113 correct |
| **Confusion matrix** | `[[38, 4], [3, 68]]` |
| **Malignant** — Precision / Recall / F1 | 0.927 / 0.905 / 0.916 |
| **Benign** — Precision / Recall / F1 | 0.944 / 0.958 / 0.951 |
| **Macro average** — Precision / Recall / F1 | 0.936 / 0.931 / 0.933 |
| **Feature scaling needed?** | **No** — 0.938 unscaled vs 0.929 scaled |
| **sklearn GaussianNB agreement** | **100.0%** — theta, var, class_prior and predict_proba all match (`np.allclose`) |
| **Sanity test** | ✅ Passes by hand calculation |

---

## 🧠 The Idea in One Paragraph

Gaussian Naive Bayes is a **generative** classifier. For each class it learns a Gaussian curve for every feature — a mean and a variance — plus how common that class is (the prior). To classify a new sample it asks, for each class: *“How likely is this sample under this class's Gaussians, times how common this class is?”* It works in **log space** to avoid underflow, and picks the class with the largest log score:

$$P(y|x) = \frac{P(x|y)\,P(y)}{P(x)}, \qquad \hat{y} = \arg\max_c \left[ \log P(c) + \sum_{j=1}^{d} \log P(x_j|c) \right]$$

$$P(x_j|c) = \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\left(-\frac{(x_j-\mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

The “naive” part is the assumption that features are **independent given the class**, so the likelihood splits into a product. That assumption is usually false — and Task 4 shows exactly what happens when you break it on purpose.

---

## 📁 Repository Contents

| File | Description |
|---|---|
| `GNB_20242023_20240523.ipynb` | **The assignment notebook** — all 7 tasks, runs top to bottom with 0 errors |
| `GNB_20242023_20240523.pdf` | **The 4-page report** — figures, tables and Task 7 answers, with properly rendered equations |
| `README.md` | This file |

---

## 🗂️ What the Notebook Does — Task by Task

### Task 1 — Data Preparation
- Loads the **Breast Cancer Wisconsin** dataset: 569 samples, 30 features, 2 classes (212 malignant, 357 benign).
- Implements our own stratified `train_test_split(x, y, test_size=0.2, seed=42)` — no sklearn split.
- Train: (456, 30) — 170 malignant / 286 benign. Test: (113, 30) — 42 / 71.
- **Scaling test:** 0.938 without scaling vs 0.929 with scaling. **GNB does not need feature scaling**, because each feature has its own mean and variance per class.

### Task 2 — GaussianNaiveBayes from Scratch
- `fit`, `predict` and `predict_proba`, all in log space, vectorised with NumPy — **no loop over samples in predict**.
- Variance smoothing: `epsilon = var_smoothing × max variance`, `var_smoothing = 1e-9`.
- **Sanity test (hand-calculated):** tiny data `[1, 2, 3]` / `[5, 6, 7]` → mean 2 and 6, variance 0.667, prior 0.5. Prediction for 2.5 is `[0]`, probability `[[0.9999, 0.0001]]`. ✅ Matches the calculator exactly.

### Task 3 — Evaluation Metrics from Scratch
- Our own `accuracy`, `confusion_matrix` and `precision_recall_f1` — no sklearn metrics. Division by zero returns 0.
- Confusion matrix heat-map, per-class and macro-average precision / recall / F1 (see table above).

### Task 4 — Experiments: Breaking the Naive Assumption
- Finds the most informative single feature — **worst concave points** (0.929 accuracy alone) — then duplicates it with **total copies 1, 3, 5, 10**.
- Accuracy stays flat at 0.929 for 1 to 10 copies, but confidence peaks at 0.999 (3–5 copies) and then falls back to 0.993 (10 copies). Confidence and accuracy do **not** move together.
- **Sanity-data version:** duplicating the single sanity feature keeps the prediction at class 0, but its probability rises from 0.9999 to 1.0 — overconfidence with no new information.

### Task 5 — Visualisation
- Fitted Gaussian curves for every class, one figure per feature (30 figures in the notebook), with training points as dots on the x-axis.
- Decision regions on a 2-feature mesh grid, with test points on top. Boundaries are **curved, not straight**, because the Gaussian equation contains quadratic terms that do not cancel when class variances differ.

### Task 6 — Comparison with scikit-learn
- sklearn `GaussianNB(var_smoothing=1e-9)` on the same data.
- `np.allclose` — theta ✅, var ✅, class_prior ✅, predict_proba ✅.
- **Prediction agreement: 100.0%.** Our implementation is mathematically identical to sklearn's.

### Task 7 — Short Questions
Five questions, each answered in 3–5 sentences, with proper equations:
1. Why “naive”? — and a real example where independence is false.
2. Why can we ignore $P(x)$?
3. What happens with zero variance and no smoothing?
4. How is GNB related to QDA?
5. Time complexity: training $O(nd)$, predicting one sample $O(kd)$.

### Conclusion
GNB achieves 93.8% accuracy, passes the hand test, and matches sklearn exactly. It needs no scaling, and duplicating a feature makes it more confident without making it more accurate — because it breaks the independence assumption.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ahmedtolba481/Gaussian-Naive-Bayes-From-Scratch.git
cd Gaussian-Naive-Bayes-From-Scratch

# 2. Install the only dependencies
pip install numpy matplotlib scikit-learn jupyter

# 3. Open the notebook and run it top to bottom
jupyter notebook GNB_20242023_20240523.ipynb
```

> The notebook runs top to bottom with **0 errors**. The only sklearn imports are `load_breast_cancer` (to load the data, Task 1) and `GaussianNB` (for the comparison, Task 6 only). No sklearn metrics, no sklearn split, no ready-made classifier in our own code.

---

## 👥 Team

| Member | ID | Role |
|---|---|---|
| **Ahmed Mohamed Elsayed Tolba** | 20242023 | Person 1 — Tasks 1, 2, 6, notebook integration |
| **Mohamed Faisal Mohamed Salem** | 20240523 | Person 2 — Tasks 3, 4, 5, 7, PDF report |

**Course:** Theoretical Foundations of Machine Learning, FCAI — Cairo University  
**Dataset:** Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer`)  
**Language:** Python 3, NumPy

---

*Built to be understood, not just to run — every line is explained, every number is checked by hand, and every assumption is tested.*
