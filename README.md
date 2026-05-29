# Manual SVM Digits Classification

A mathematical implementation of a **Multi-Class Support Vector Machine (SVM) with a Linear Kernel** built completely from scratch. This project avoids high-level machine learning libraries (such as scikit-learn) to focus entirely on foundational mathematical programming, optimization theory, and vectorization techniques.

## SOAR Framework Overview

### Situation
As part of the advanced Machine Learning curriculum (INF372) at the University of Yaoundé I, the task was to build a robust classifier capable of recognizing handwritten digits ($0$ to $9$) from a subset of the MNIST dataset. The training pipeline utilizes $3,000$ normalized $32\times32$ grayscale images.

### Obstacle
Modern ML pipelines obscure structural mechanics behind abstract library calls (e.g., `model.fit()`). The core challenge of this project was to implement multi-class classification ($10$ separate targets) under rigid low-resource optimization constraints without relying on external machine learning frameworks. This required manual formulation of convex quadratic programming problems, explicit Gram Matrix calculation, and manual implementation of a multi-class strategy.

### Action
The implementation was executed through a rigorous mathematical translation workflow using strictly NumPy and a convex optimization modeling framework (`cvxpy`):
1. **Mathematical Data Preprocessing:** Flattened spatial matrices into structural feature vectors $x_i \in \mathbb{R}^{1024}$ and mapped classes using explicit labels.
2. **Gram Matrix Optimization:** Computed the explicit inner-product matrix $G_{ij} = \langle x_i, x_j \rangle = x_i^T x_j$ to form the core kernel surface.
3. **Convex Optimization Solver:** Formulated and solved the dual quadratic programming problem:
   $$\min_{\lambda} \frac{1}{2} \lambda^T Q \lambda - e^T \lambda \quad \text{subject to} \quad 0 \le \lambda_i \le C, \,\, \sum \lambda_i y_i = 0$$
4. **Vectorized Support Vector Extraction:** Filtered optimal Lagrange multipliers ($\lambda_i > 0$) to extract Support Vectors, dynamically computing weights $w$ and bias $b$.
5. **Meta-Architecture Strategy:** Built a robust **One-vs-All (OvA)** meta-architecture combining $10$ distinct binary SVM decision boundaries to process the multi-class prediction spaces.

### Results
* **Deterministic Metrics:** Achieved clear, stable decision boundaries across highly ambiguous handwritten samples.
* **Feature Interpretability:** Reshaping the manual weight vectors ($w$) back into the native $32\times32$ dimensions provides spatial heatmaps that clearly reveal the specific pixels and strokes the model uses to distinguish boundaries.
* **Error Analysis Mapping:** Generated raw confidence distributions ($s_k(x) = w_k^T x + b_k$) and structural confusion matrices, isolating exact morphological overlap regions (e.g., distinguishing features between $1$/$7$ and $3$/$8$).

---

## Features
* **Zero ML Libraries:** Constructed entirely with Python, NumPy, and CVXPY.
* **Convex Optimization:** Direct minimization of the dual SVM formulation.
* **Hyperparameter Control:** Scalable regularization boundary control via parameter $C$.
* **OvA Architecture:** Scalable multi-class decision mapping.

## Dataset Source
The framework is trained and evaluated on a customized, normalized subset of handwritten digits ($0$ to $9$) based on the foundational MNIST dataset structure:
* **Format:** Grayscale PNG arrays
* **Resolution:** $32 \times 32$ pixels (yielding an explicit structural feature vector $x_i \in \mathbb{R}^{1024}$)
* **Scope:** 3,000 training samples optimized for explicit matrix constraints.

## Getting Started

### Prerequisites
* Python 3.8 or higher
* NumPy
* CVXPY (Convex optimization modeling framework)
* Matplotlib (For distribution and error visualization)
* Jupyter Notebook

### Installation
Clone the repository and install the structural dependencies:
```bash
git clone [https://github.com/KamdemSamuel/Convex-Optimization-SVM-Scratch.git](https://github.com/KamdemSamuel/Convex-Optimization-SVM-Scratch.git)
cd Convex-Optimization-SVM-Scratch
pip install numpy cvxpy matplotlib
Usage
Run the interactive workspace notebook to view the mathematical proofs, execution stages, and visualization layers:

Bash
jupyter notebook Manual_SVM_Digits_Classification.ipynb
Repository Structure
Plaintext
Convex-Optimization-SVM-Scratch/
├── Manual_SVM_Digits_Classification.ipynb  # Main structural math and code implementation
├── slide_TP5.pdf                           # Academic presentation and analytical slides
├── README.md                               # Project documentation
└── LICENSE                                 # License file
Authors
Tchuenche Kamdem Samuel Yedidya - LinkedIn


License
This project is licensed under the MIT License.
