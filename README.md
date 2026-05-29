# Manual SVM Digits Classification

A mathematical implementation of a **Multi-Class Support Vector Machine (SVM) with a Linear Kernel** built completely from scratch. This project avoids high-level machine learning libraries (such as scikit-learn) to provide deep insight into the mechanics of convex optimization and kernel methods.

## SOAR Framework Overview

### Situation
As part of the advanced Machine Learning curriculum (INF372) at the University of Yaoundé I, the task was to build a robust classifier capable of recognizing handwritten digits ($0$ to $9$) from a preprocessed MNIST dataset. The constraint was to implement everything from mathematical first principles without relying on pre-built ML frameworks.

### Obstacle
Modern ML pipelines obscure structural mechanics behind abstract library calls (e.g., `model.fit()`). The core challenge of this project was to implement multi-class classification ($10$ separate binary decision boundaries) using explicit convex optimization, requiring manual implementation of the dual SVM formulation, Gram matrix construction, and Lagrange multiplier extraction.

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
* **Feature Interpretability:** Reshaping the manual weight vectors ($w$) back into the native $32\times32$ dimensions provides spatial heatmaps that clearly reveal the specific pixels and strokes required to distinguish each digit.
* **Error Analysis Mapping:** Generated raw confidence distributions ($s_k(x) = w_k^T x + b_k$) and structural confusion matrices, isolating exact morphological overlap regions (e.g., distinguishing 3 from 8).

---

## Features
* **Zero ML Libraries:** Constructed entirely with Python, NumPy, and CVXPY.
* **Convex Optimization:** Direct minimization of the dual SVM formulation.
* **Hyperparameter Control:** Scalable regularization boundary control via parameter $C$.
* **OvA Architecture:** Scalable multi-class decision mapping.

## Dataset Details

### Source
The framework is trained and evaluated on a customized, normalized subset of handwritten digits ($0$ to $9$) based on the foundational MNIST dataset structure.

### Specifications
* **Format:** Grayscale PNG arrays
* **Resolution:** $32 \times 32$ pixels (yielding an explicit structural feature vector $x_i \in \mathbb{R}^{1024}$)
* **Total Samples:** 3,000 training samples optimized for explicit matrix constraints
* **Class Distribution:** Balanced across 10 digit classes (0-9)
* **Preprocessing:** Images normalized to [0, 1] range and flattened to 1024-dimensional vectors

### Dataset Download
The MNIST dataset can be obtained from:
- **Official source:** http://yann.lecun.com/exdb/mnist/
- **Via Python (scikit-learn):**
  ```python
  from sklearn.datasets import fetch_openml
  mnist = fetch_openml('mnist_784', version=1)
  X, y = mnist.data, mnist.target
  ```
- **Via TensorFlow/Keras:**
  ```python
  from tensorflow.keras.datasets import mnist
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  ```

### Expected File Structure
```
data/
├── train/
│   ├── digit_0/
│   │   ├── 0_0001.png
│   │   ├── 0_0002.png
│   │   └── ...
│   ├── digit_1/
│   └── ...
└── test/
    ├── digit_0/
    └── ...
```

---

## Getting Started

### Prerequisites
* Python 3.8 or higher
* pip (Python package manager)
* Virtual environment (recommended)

### Installation

#### Step 1: Clone the repository
```bash
git clone https://github.com/KamdemSamuel/Convex-Optimization-SVM-Scratch.git
cd Convex-Optimization-SVM-Scratch
```

#### Step 2: Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy==1.24.3 cvxpy==1.3.2 matplotlib==3.7.1 jupyter==1.0.0
```

#### Step 4: Prepare the dataset
- Download MNIST from the sources above
- Place dataset in a `data/` directory at the project root
- Or modify the notebook path to point to your dataset location

#### Step 5: Launch Jupyter Notebook
```bash
jupyter notebook Manual_SVM_Digits_Classification.ipynb
```

### Verification
After installation, verify everything works:
```bash
python -c "import numpy; import cvxpy; import matplotlib; print('All dependencies installed successfully!')"
```

---

## Usage

Run the interactive workspace notebook to view the mathematical proofs, execution stages, and visualization layers:

```bash
jupyter notebook Manual_SVM_Digits_Classification.ipynb
```

**Notebook workflow:**
1. Data loading and preprocessing (flattening, normalization)
2. Gram matrix computation
3. Dual QP formulation and CVXPY solver setup
4. Support vector extraction and weight computation
5. One-vs-All classifier training (10 binary SVMs)
6. Prediction on test set
7. Visualization: decision boundaries, weight heatmaps, confusion matrices

---

## Repository Structure
```
Convex-Optimization-SVM-Scratch/
├── Manual_SVM_Digits_Classification.ipynb  # Main structural math and code implementation
├── slide_TP5.pdf                           # Academic presentation and analytical slides
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── LICENSE                                 # MIT License
└── data/                                   # Dataset directory (create as needed)
```

---

## Performance Metrics
- **Training Set Accuracy:** ~96-98% (depending on regularization parameter C)
- **Test Set Accuracy:** ~94-96%
- **Support Vectors:** ~40-50% of training set (sparse solution due to convex formulation)
- **Training Time:** ~5-10 minutes for 3,000 samples (depends on hardware)

---

## Mathematical Background

### Dual SVM Formulation
Given labeled data $(x_i, y_i)$ where $y_i \in \{-1, +1\}$, the dual optimization problem is:
$$\max_{\lambda} \sum_{i=1}^n \lambda_i - \frac{1}{2} \sum_{i,j} \lambda_i \lambda_j y_i y_j K(x_i, x_j)$$

subject to: $0 \le \lambda_i \le C$ and $\sum_i \lambda_i y_i = 0$

### One-vs-All Strategy
For $K$ classes, train $K$ independent binary classifiers:
- Classifier $k$: positive class = digit $k$, negative class = all other digits
- Final prediction: $\arg\max_k s_k(x)$ where $s_k(x) = w_k^T x + b_k$

---

## References
- Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*
- Boyd & Vandenberghe (2004). *Convex Optimization*
- CVXPY Documentation: https://www.cvxpy.org/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/

---

## Authors
[Tchuenche Kamdem Samuel Yedidya](https://www.linkedin.com/in/kamdem-samuel-yedidya/) - LinkedIn

**Course:** INF372 — Machine Learning, Université de Yaoundé I

---

## License
This project is licensed under the MIT License. See `LICENSE` file for details.
