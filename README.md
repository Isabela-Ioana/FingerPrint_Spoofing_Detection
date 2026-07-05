# Fingerprint Spoofing Detection

A Machine Learning pipeline built entirely **from scratch using Python, NumPy, and SciPy**. This project focuses on binary classification to identify genuine vs. counterfeit (spoofed) fingerprint images using a 6-dimensional high-level feature dataset that capture the subtle structural anomalies of synthetic replication materials.

To demonstrate deep theoretical understanding, standard ML libraries (like *scikit-learn*) were deliberately omitted; data splitting, metrics evaluation, dimensionality reduction, and generative classification are all implemented manually.




## Project Architecture & Modules

*   `mvg.py` – **Multivariate Gaussian Classifier (MVG)**: A generative model implemented using 100% vectorization and broadcasting to eliminate slow python loops.
*   `pca.py` – **Principal Component Analysis (PCA)**: Unsupervised feature extraction using Singular Value Decomposition (SVD).
*   `lda.py` – **Linear Discriminant Analysis (LDA)**: Supervised dimensionality reduction via generalized eigenvalue problem optimization ($S_B w = \lambda S_W w$).
*   `helping_functions.py` – Utilities for dataset loading, automated random shuffling & partitioning (`train_test_split`), and custom evaluation matrix generation.
*   `main.py` – The main pipeline execution script evaluating and comparing the three core architectural scenarios.



## Benchmarks & Performance Summary

The model architecture was trained on 70% of the data (4,200 samples) and validated on an unseen 30% split (1,800 samples). 

### Model Performance Comparison

| Pipeline / Model | Input Dimensions | Accuracy | minDCF ($P_{true}=0.5$) | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw MVG (Baseline)** | 6D | **91.56%** | **0.1628** | **79** | **73** |
| **PCA + MVG** | 2D | **90.00%** | **0.1985** | **83** | **97** |
| **LDA + MVG** | 1D | **89.50%** | **0.2004** | **92** | **97** |



## Theoretical & Engineering Insights


*   **PCA Efficiency:** Compressing the feature space by **66%** (moving from 6D down to a 2D PCA subspace) resulted in a minor accuracy drop of only **1.56%**. This proves that the custom SVD implementation successfully captured the highest directions of variance while scrubbing away redundant noise.
*   **Supervised LDA in 1D:** For a binary setup, LDA is mathematically limited to a single projection dimension ($C - 1 = 1$). Projecting all 6 original characteristics onto a **single line** still yielded an outstanding **89.50% accuracy**, demonstrating massive class separability along the computed discriminant vector.
*   **Strictly Avoiding Data Leakage:** The `PCA` and `LDA` modules store internal parameters ($\mu$, $P$, $w$) exclusively during the `.fit()` stage on the training data. When `.transform()` is called on the validation set, it applies the exact statistical parameters learned from the training split, keeping testing metrics completely un-compromised.


## Mathematical Foundations

Here are the mathematical core principles implemented from scratch using matrix algebra in NumPy and SciPy:

### 1. Data Centering & Covariance
Before extraction, the features dataset $X \in \mathbb{R}^{d \times N}$ (where $d=6$ features, $N=4200$ training samples) is centered using its empirical mean vector $\mu$:

$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i, \quad \quad \Sigma = \frac{1}{N} (X - \mu)(X - \mu)^T$$

### 2. Principal Component Analysis (PCA) via SVD
Instead of a direct Eigendecomposition, the centered data matrix is decomposed using **Singular Value Decomposition (SVD)** for numerical stability:

$$X_{centered} = U \cdot S \cdot V^T$$

The first $k=2$ columns of the left singular vectors matrix $U$ are chosen to form the projection matrix $P = U_{:, :k}$. The reduced representation is computed as:

$$X_{pca} = P^T \cdot X_{centered}$$

### 3. Linear Discriminant Analysis (LDA)
LDA maximizes the class separability by finding a projection vector $w$ that optimizes the Fisher criterion. It computes the *Within-class scatter matrix* ($S_W$) and *Between-class scatter matrix* ($S_B$):

$$S_W = \sum_{c \in \{0,1\}} \sum_{x \in X_c} (x - \mu_c)(x - \mu_c)^T$$

$$S_B = \sum_{c \in \{0,1\}} N_c (\mu_c - \mu)(\mu_c - \mu)^T$$

The projection vector $w$ is found by solving the **Generalized Eigenvalue Problem** using `scipy.linalg.eigh`:

$$S_B w = \lambda S_W w$$

### 4. Multivariate Gaussian Classifier & Log-Likelihood Ratio (LLR) Test
Instead of computing raw probability densities—which causes numerical underflow due to extremely small values—the model operates entirely in the log-probability domain. 

For each sample $x$, the model utilizes the class-specific mean vector $\mu_c$ and covariance matrix $\Sigma_c$ to calculate the log-density for both the Spoofed ($c=0$) and Authentic ($c=1$) classes:

$$\ln f(x \ | \ \mu_c, \Sigma_c) = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\Sigma_c| - \frac{1}{2}(x - \mu_c)^T \Sigma_c^{-1} (x - \mu_c)$$

The final binary classification decision is formulated as a **Log-Likelihood Ratio (LLR) Test**. The classifier evaluates whether the log-density of the sample being authentic outweighs the log-density of it being spoofed:

$$LLR(x) = \ln f(x \ | \ \mu_1, \Sigma_1) - \ln f(x \ | \ \mu_0, \Sigma_0)$$

The decision rule maps the continuous LLR score into a discrete prediction using an implicit threshold of $0$:

$$\text{Prediction} = \begin{cases} 1 \quad \text{(Authentic),} & \text{if } LLR(x) > 0 \\ 0 \quad \text{(Spoofed),} & \text{if } LLR(x) \le 0 \end{cases}$$

In the custom NumPy implementation, this vectorized comparison is executed elegantly and efficiently across all samples simultaneously using `np.argmax(log_densities, axis=0)`.

The continuous LLR score is further leveraged to evaluate the system under a strict risk-minimization framework using the **Minimum Detection Cost Function (minDCF)**. Instead of assessing unweighted accuracy, minDCF incorporates real-world security costs ($C_{FN}$ and $C_{FP}$) and the prior probability of encountering an authentic user ($P_{true}$):

$$DCF = C_{FN} \times P_{FN} \times P_{true} + C_{FP} \times P_{FP} \times (1 - P_{true})$$

The custom implementation sweeps all possible operational thresholds across the sorted validation LLR spectrum to find the absolute minimum cost, which is then normalized against a dummy (no-information) classifier:

$$\text{minDCF}_{normalized} = \frac{\min_{\tau} \text{DCF}(\tau)}{\min(C_{FN} P_{true}, C_{FP} (1 - P_{true}))}$$

* $\min_{\tau} \text{DCF}(\tau)$: The **best (lowest) cost achieved by the model**. The code sweeps all possible thresholds ($\tau$) across the LLR scores to find where the model makes the fewest and least dangerous mistakes.

* $\min(C_{FN} P_{true}, C_{FP} (1 - P_{true}))$: **It represents the cost of making the safest "blind" decision**. The system automatically picks whichever option causes the least damage:
  * *Reject All*: We risk only annoying real users (costs $C_{FN} \times P_{true}$).
  * *Accept All*: We risk letting all attackers in (costs $C_{FP} \times (1 - P_{true})$).
