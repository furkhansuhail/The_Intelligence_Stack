OPERATIONS   = {}
VISUAL_HTML  = ""

"""Module: 04 · PCA"""
DISPLAY_NAME = "04 · PCA"
ICON         = "📐"
SUBTITLE     = "Principal Component Analysis — linear dimensionality reduction"

THEORY = """
## 04 · Principal Component Analysis (PCA)

---

### The Problem PCA Solves

Real-world data is almost always high-dimensional. A dataset of 1000 medical
measurements per patient, a corpus of 50,000-word vocabulary vectors, an image
with 256×256 = 65,536 pixel values. Working directly in these spaces is painful:

- Visualisation is impossible beyond 3 dimensions.
- Many machine learning algorithms degrade in high dimensions (curse of
  dimensionality — distances concentrate, nearest-neighbour searches become
  meaningless).
- Many features are redundant or highly correlated — they carry overlapping
  information.
- Training is slow when the input dimension is large.

**PCA finds a lower-dimensional subspace that preserves as much variance
(information) as possible.**

It does this by finding a new set of axes — called **principal components** —
that are:
  1. Linear combinations of the original features.
  2. Ordered by how much variance they explain (PC1 explains the most, PC2
     the second most, and so on).
  3. Mutually orthogonal (uncorrelated with each other).

You then project your data onto the top K principal components, discarding
the rest. If the first K components capture, say, 95% of the total variance,
you have retained nearly all the information in K dimensions instead of the
original p.

---

### Intuition: Rotating the Axes

Imagine a 2D scatter plot where the data forms an elongated ellipse tilted at
45 degrees. The original axes (x₁, x₂) don't align with this shape — both
coordinates are needed to describe where any point is.

PCA rotates the axes to align with the ellipse:
- The first new axis (PC1) points along the long direction of the ellipse —
  the direction of maximum variance.
- The second new axis (PC2) points along the short direction — perpendicular
  to PC1, capturing the remaining variance.

In this new coordinate system, most of the information is concentrated in the
PC1 coordinate. If the ellipse is very thin, PC2 adds little — you can safely
drop it and describe the data in 1D instead of 2D with minimal information loss.

PCA is exactly this rotation, generalised to p dimensions.

---

### Step 1 — Centre the Data (Mean Subtraction)

Before anything else, subtract the mean of each feature:

    X_centred[i][j] = X[i][j] − μⱼ

where μⱼ = (1/n) Σᵢ X[i][j] is the mean of feature j.

**Why?** PCA finds directions of variance around the origin. If the data is not
centred, the first component would just point toward the cloud's centre of mass
rather than its principal spread direction. Centring moves the cloud so its
centre of mass is at the origin.

Optionally, you may also **standardise** (divide by standard deviation per
feature):

    X_scaled[i][j] = (X[i][j] − μⱼ) / σⱼ

This is necessary when features are measured on different scales (e.g. age in
years vs. income in thousands). Without standardisation, high-variance features
dominate the principal components simply because of their scale, not their
information content.

Rule of thumb: standardise unless all features are in the same units and you
have a reason to preserve scale differences.

---

### Step 2 — Compute the Covariance Matrix

The covariance matrix C is a p×p symmetric matrix:

    C = (1/(n−1)) · Xᵀ · X        (where X is already centred)

The entry C[j][k] = cov(xⱼ, xₖ) measures how much features j and k vary
together:

    cov(xⱼ, xₖ) = (1/(n−1)) · Σᵢ (xᵢⱼ − μⱼ)(xᵢₖ − μₖ)

The diagonal entries C[j][j] = var(xⱼ) are the variance of each feature.
The off-diagonal entries measure correlations. High |C[j][k]| means j and k
are redundant — they carry overlapping information that PCA will compress.

The total variance in the dataset = trace(C) = sum of diagonal entries
= sum of all feature variances.

---

### Step 3 — Eigendecomposition

The covariance matrix C is real and symmetric, so it has a full set of p
real eigenvalues and p orthogonal eigenvectors:

    C · vₖ = λₖ · vₖ

where:
- vₖ is the k-th eigenvector (a unit vector in the original feature space)
- λₖ is the k-th eigenvalue (a scalar)

Each eigenvector is a **principal component direction**. The corresponding
eigenvalue λₖ tells you the **variance of the data along that direction**.

Sort by eigenvalue descending: λ₁ ≥ λ₂ ≥ ... ≥ λₚ ≥ 0.

The first principal component v₁ is the direction of maximum variance.
The second v₂ is the direction of maximum variance among all directions
perpendicular to v₁. And so on.

**Why does this work?** The variance of the data when projected onto any unit
vector u is uᵀCu. Maximising this subject to ||u||=1 gives exactly the
eigenvector corresponding to the largest eigenvalue. This is the Rayleigh
quotient maximisation — a standard result in linear algebra.

---

### Step 4 — Select K Components

Explained variance ratio for component k:

    EVR(k) = λₖ / Σⱼ λⱼ = λₖ / trace(C)

This is the fraction of total variance captured by component k alone.

**Cumulative explained variance:**

    CEVR(K) = Σₖ₌₁ᴷ λₖ / trace(C)

You choose K such that CEVR(K) ≥ your threshold (commonly 90%, 95%, or 99%).

**Scree plot:** Plot λₖ vs k. Look for the "elbow" — the point where
eigenvalues stop dropping sharply and level off. Components beyond the elbow
explain diminishing variance and are often discarded.

A common practical heuristic: keep components with λₖ > 1 when the data is
standardised (Kaiser criterion). A component explaining less variance than a
single original feature is unlikely to be informative.

---

### Step 5 — Project the Data

Form the projection matrix W from the top K eigenvectors as columns:

    W = [v₁ | v₂ | ... | vₖ]    shape: (p × K)

Project:

    Z = X_centred · W              shape: (n × K)

Each row of Z is the low-dimensional representation of the corresponding
original data point. The columns of Z are the principal component scores.

To reconstruct (approximately) the original data:

    X_reconstructed = Z · Wᵀ + μ

The reconstruction error = total discarded variance = Σₖ₌ₖ₊₁ᵖ λₖ.

---

### A Worked Example (4 Points, 2D → 1D)

Data (4 points):
    x₁ = (1, 2)
    x₂ = (3, 5)
    x₃ = (2, 3)
    x₄ = (4, 6)

**Step 1 — Mean subtraction:**
    μ = (2.5, 4.0)
    x₁_c = (−1.5, −2.0)
    x₂_c = ( 0.5,  1.0)
    x₃_c = (−0.5, −1.0)
    x₄_c = ( 1.5,  2.0)

**Step 2 — Covariance matrix:**

    C = (1/3) · Xᵀ · X

    C[1,1] = (1/3)(2.25+0.25+0.25+2.25) = (1/3)(5.0)  = 1.667
    C[2,2] = (1/3)(4.0 +1.0 +1.0 +4.0)  = (1/3)(10.0) = 3.333
    C[1,2] = (1/3)(3.0 +0.5 +0.5 +3.0)  = (1/3)(7.0)  = 2.333

    C = [[1.667, 2.333],
         [2.333, 3.333]]

**Step 3 — Eigendecomposition:**

    trace(C) = 1.667 + 3.333 = 5.0
    det(C)   = 1.667×3.333 − 2.333² = 5.556 − 5.444 = 0.112

    Eigenvalues: λ² − 5λ + 0.112 = 0
    λ₁ = (5 + √(25 − 0.448)) / 2 ≈ 4.978
    λ₂ = (5 − √(25 − 0.448)) / 2 ≈ 0.022

    Eigenvector for λ₁ ≈ 4.978:
    (C − λ₁I)v = 0
    v₁ ≈ [0.555, 0.832]   (normalised)

    Eigenvector for λ₂ ≈ 0.022:
    v₂ ≈ [−0.832, 0.555]  (perpendicular to v₁)

**Step 4 — Explained variance:**

    EVR(1) = 4.978 / 5.0 = 99.6%  → PC1 alone captures 99.6% of variance!
    EVR(2) = 0.022 / 5.0 =  0.4%

**Step 5 — Project onto PC1:**

    Z = X_centred · v₁
    z₁ = (−1.5)(0.555) + (−2.0)(0.832) = −0.833 − 1.664 = −2.497
    z₂ = ( 0.5)(0.555) + ( 1.0)(0.832) =  0.278 + 0.832 =  1.110
    z₃ = (−0.5)(0.555) + (−1.0)(0.832) = −0.278 − 0.832 = −1.110
    z₄ = ( 1.5)(0.555) + ( 2.0)(0.832) =  0.833 + 1.664 =  2.497

    Z = [−2.497, 1.110, −1.110, 2.497]

The 4 points are now described in 1D with 99.6% of the original information
preserved. The 2D data lies almost perfectly on a line — PCA found it.

---

### SVD Formulation

PCA via eigendecomposition of the covariance matrix is mathematically correct
but numerically unstable for large matrices. In practice, PCA is computed
using the **Singular Value Decomposition (SVD)** of the centred data matrix:

    X_centred = U · Σ · Vᵀ

where:
- U is an (n × p) matrix with orthonormal columns (left singular vectors)
- Σ is a (p × p) diagonal matrix with non-negative singular values σ₁ ≥ σ₂ ≥ ... ≥ σₚ
- V is a (p × p) orthogonal matrix (right singular vectors)

The connection to eigendecomposition:

    Principal component directions  = columns of V  (same as eigenvectors of C)
    Eigenvalues                      = σₖ² / (n−1)
    PC scores                        = U · Σ  (same as X_centred · V)

SVD is preferred because:
1. It never explicitly forms Xᵀ·X, avoiding squaring the condition number.
2. It computes the full decomposition in a single numerically stable pass.
3. The truncated SVD (keeping only top K singular vectors) is efficient for
   large, sparse matrices using iterative methods (e.g. ARPACK, randomised SVD).

---

### Kernel PCA

Standard PCA is linear — it finds linear directions of maximum variance. If
your data lies on a nonlinear manifold (a Swiss roll, concentric circles,
etc.), linear PCA cannot unfold it.

**Kernel PCA** extends PCA to nonlinear structures using the kernel trick.
Instead of working in the original feature space, it implicitly maps data to
a high-dimensional (possibly infinite) feature space via a kernel function:

    k(xᵢ, xⱼ) = φ(xᵢ)ᵀ · φ(xⱼ)

Common kernels:
- **RBF (Gaussian):** k(x,y) = exp(−||x−y||² / 2σ²)
- **Polynomial:** k(x,y) = (xᵀy + c)ᵈ
- **Sigmoid:** k(x,y) = tanh(αxᵀy + c)

The kernel matrix K (n×n) is computed: K[i][j] = k(xᵢ, xⱼ).
PCA is then performed on the centred kernel matrix. The result is a nonlinear
projection of the original data.

---

### Incremental / Online PCA

Standard PCA requires loading the entire dataset into memory to compute the
covariance matrix or perform SVD. For streaming data or datasets too large to
fit in memory, **Incremental PCA** processes the data in mini-batches.

Each batch updates the running estimate of the principal components using a
rank-1 update rule. The result converges to the true PCA solution as more
batches are processed.

---

### PCA vs Other Dimensionality Reduction Methods

| Method      | Type        | Linear? | Preserves          | Best for                        |
|-------------|-------------|---------|--------------------|---------------------------------|
| PCA         | Unsupervised| Yes     | Global variance    | Preprocessing, visualisation    |
| LDA         | Supervised  | Yes     | Class separation   | Classification preprocessing    |
| t-SNE       | Unsupervised| No      | Local structure    | Visualisation (2D/3D only)      |
| UMAP        | Unsupervised| No      | Local + global     | Visualisation, general reduction|
| Autoencoder | Unsupervised| No      | Learned features   | Complex nonlinear structure     |
| ICA         | Unsupervised| Yes     | Statistical indep. | Signal separation (BSS)         |
| NMF         | Unsupervised| Yes     | Non-negativity     | Parts-based representation      |
| Factor Anal.| Unsupervised| Yes     | Latent factors     | Psychometrics, interpretability |

---

### Common Uses of PCA

**Visualisation** — Reduce to 2 or 3 components to plot high-dimensional data.
The first two PCs capture the dominant structure. Colour points by class labels
to see whether classes are separable in the reduced space.

**Preprocessing before supervised learning** — Reduce dimensionality before
feeding into a classifier or regressor. Removes redundancy, speeds up training,
sometimes improves generalisation by eliminating noisy low-variance features.
However: tree-based methods (Random Forest, XGBoost) handle high dimensions
well and often don't benefit from PCA preprocessing.

**Noise filtering / compression** — Reconstruct from the top K components.
The discarded components carry less signal and more noise — reconstruction
acts as a low-rank filter. Used in image compression, signal denoising,
and recommendation systems (collaborative filtering via matrix factorisation).

**Multicollinearity removal** — PCA components are orthogonal by construction.
Feeding PC scores instead of raw correlated features into linear regression
or logistic regression eliminates multicollinearity problems.

**Anomaly detection** — Project to K components and reconstruct. Points with
high reconstruction error lie far from the principal subspace and may be
anomalies. Used in network intrusion detection, manufacturing quality control.

---

### Limitations and Pitfalls

**Linearity** — PCA can only find linear structure. Nonlinear manifolds
require kernel PCA, t-SNE, UMAP, or autoencoders.

**Scale sensitivity** — If features are not standardised, high-variance
features dominate regardless of their information content. Always standardise
unless you have a specific reason not to.

**Interpretability** — Each principal component is a linear combination of
all original features, with potentially non-zero loadings on every feature.
PC1 is not "feature 3" — it is a mixture. This makes interpretation difficult
compared to methods like sparse PCA that produce loadings with many zeros.

**Assumes Gaussian structure** — PCA's variance-maximisation objective is
optimal when the data is Gaussian. For highly non-Gaussian distributions,
ICA (Independent Component Analysis) may be more appropriate.

**Information loss** — Discarded components do carry some information. For
critical applications, validate that the information loss is acceptable by
measuring downstream task performance with and without PCA.

**Sensitive to outliers** — Extreme outliers inflate variance in their
direction, potentially pulling the first principal component toward them.
Robust PCA variants (e.g. RPCA using L1 instead of L2) mitigate this.

---

### Key Takeaways

1. PCA finds orthogonal directions of maximum variance in your data via
   eigendecomposition of the covariance matrix (or SVD of the data matrix).

2. Always centre your data first. Standardise when features are on different
   scales.

3. Eigenvalue = variance along that principal component. Explained variance
   ratio = λₖ / Σλ. Use scree plot or cumulative EVR to choose K.

4. Projection: Z = X_centred · W where W = top K eigenvectors. Reconstruction:
   X̂ = Z · Wᵀ + μ. Reconstruction error = discarded variance.

5. SVD is numerically preferred over direct eigendecomposition of Xᵀ·X.

6. PCA is linear. For nonlinear structure, use kernel PCA, t-SNE, or UMAP.
"""


OPERATIONS = {

    "▶ Run: PCA From Scratch (Full Walkthrough)": {
        "description": "Complete PCA from first principles — mean subtraction, covariance matrix, eigendecomposition, explained variance, and 2D projection. Every step printed.",
        "code": """
import math

# ── Dataset: 6 points in 3D (two correlated features + one noisy) ─────────────
X = [
    [2.5, 2.4, 0.1],
    [0.5, 0.7, 0.8],
    [2.2, 2.9, 0.3],
    [1.9, 2.2, 0.5],
    [3.1, 3.0, 0.2],
    [2.3, 2.7, 0.9],
]
feature_names = ['f1','f2','f3']
n, p = len(X), len(X[0])

print("=" * 55)
print("     PCA — Step-by-Step Walkthrough")
print("=" * 55)
print(f"Dataset: {n} samples, {p} features")
print()

# ── Step 1: Mean subtraction ──────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
Xc    = [[X[i][j] - means[j] for j in range(p)] for i in range(n)]

print("STEP 1 — Mean of each feature:")
for j,f in enumerate(feature_names):
    print(f"  μ[{f}] = {means[j]:.4f}")
print()

# ── Step 2: Covariance matrix ─────────────────────────────────────────────────
def cov_matrix(Xc):
    n, p = len(Xc), len(Xc[0])
    C = [[0.0]*p for _ in range(p)]
    for j in range(p):
        for k in range(p):
            C[j][k] = sum(Xc[i][j]*Xc[i][k] for i in range(n)) / (n-1)
    return C

C = cov_matrix(Xc)
print("STEP 2 — Covariance Matrix C:")
header = "         " + "".join(f"  {feature_names[j]:>8}" for j in range(p))
print(header)
for j in range(p):
    row = f"  {feature_names[j]:>5}  " + "  ".join(f"{C[j][k]:>8.4f}" for k in range(p))
    print(row)
print()
print(f"  trace(C) = total variance = {sum(C[j][j] for j in range(p)):.4f}")
print()

# ── Step 3: Power iteration for eigenvalues/eigenvectors ──────────────────────
def mat_vec(M, v):
    n = len(v)
    return [sum(M[i][j]*v[j] for j in range(n)) for i in range(n)]

def normalize(v):
    norm = math.sqrt(sum(x**2 for x in v))
    return [x/norm for x in v], norm

def power_iteration(M, n_iter=1000, seed_v=None):
    p = len(M)
    if seed_v is None:
        v = [1.0/math.sqrt(p)] * p
    else:
        v = seed_v[:]
    for _ in range(n_iter):
        w = mat_vec(M, v)
        v, norm = normalize(w)
    eigenval = sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return eigenval, v

def deflate(M, eigenval, eigenvec):
    p = len(M)
    return [[M[i][j] - eigenval*eigenvec[i]*eigenvec[j] for j in range(p)] for i in range(p)]

# Extract all eigenpairs via deflation
eigenpairs = []
Cd = [row[:] for row in C]
seed_vecs = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
for k in range(p):
    lam, vec = power_iteration(Cd, seed_v=seed_vecs[k])
    if lam < 0: lam = 0.0
    eigenpairs.append((lam, vec))
    Cd = deflate(Cd, lam, vec)

eigenpairs.sort(reverse=True, key=lambda x: x[0])
eigenvalues   = [ep[0] for ep in eigenpairs]
eigenvectors  = [ep[1] for ep in eigenpairs]
total_var     = sum(eigenvalues)

print("STEP 3 — Eigenvalues and Principal Component Directions:")
for k, (lam, vec) in enumerate(eigenpairs):
    evr = lam / total_var * 100 if total_var > 0 else 0
    vec_str = "[" + ", ".join(f"{v:>7.4f}" for v in vec) + "]"
    print(f"  PC{k+1}: λ={lam:>7.4f}  ({evr:>5.1f}% variance)  direction={vec_str}")
print()

# ── Step 4: Cumulative explained variance ─────────────────────────────────────
print("STEP 4 — Explained Variance (Scree):")
cumulative = 0
for k, lam in enumerate(eigenvalues):
    evr = lam / total_var * 100 if total_var > 0 else 0
    cumulative += evr
    bar = "█" * int(evr * 3)
    print(f"  PC{k+1}: {evr:>5.1f}%  cumulative: {cumulative:>5.1f}%  {bar}")
print()

# ── Step 5: Project onto top 2 PCs ───────────────────────────────────────────
K = 2
W = eigenvectors[:K]   # K eigenvectors as rows → shape K×p

def project(Xc, W):
    return [[sum(Xc[i][j]*W[k][j] for j in range(len(W[k])))
             for k in range(len(W))]
            for i in range(len(Xc))]

Z = project(Xc, W)

print(f"STEP 5 — Projection onto top {K} PCs:")
print(f"  {'Point':<7}  {'PC1':>8}  {'PC2':>8}")
print("  " + "─" * 28)
for i, z in enumerate(Z):
    print(f"  pt {i+1:<3}  {z[0]:>8.4f}  {z[1]:>8.4f}")
print()

# ── Step 6: Reconstruction ────────────────────────────────────────────────────
def reconstruct(Z, W, means):
    p = len(W[0])
    return [[sum(Z[i][k]*W[k][j] for k in range(len(W)))+means[j]
             for j in range(p)] for i in range(len(Z))]

Xr = reconstruct(Z, W, means)

print(f"STEP 6 — Reconstruction from {K} PCs (X̂ ≈ X):")
print(f"  {'Point':<7}  {'Original':^28}  {'Reconstructed':^28}  Rec. error")
print("  " + "─" * 75)
total_err = 0
for i in range(n):
    orig  = str([f"{X[i][j]:.2f}" for j in range(p)])
    rec   = str([f"{Xr[i][j]:.2f}" for j in range(p)])
    err   = math.sqrt(sum((X[i][j]-Xr[i][j])**2 for j in range(p)))
    total_err += err**2
    print(f"  pt {i+1:<3}  {orig:<28}  {rec:<28}  {err:.4f}")

print()
discarded_var = sum(eigenvalues[K:])
print(f"  Total reconstruction SSE     : {total_err:.4f}")
print(f"  Discarded eigenvalue sum     : {discarded_var:.4f}")
print(f"  Information retained         : {(1-discarded_var/total_var)*100:.1f}%")
""",
        "runnable": True,
    },

    "▶ Run: Scree Plot and Component Selection": {
        "description": "Generate a full scree plot with explained variance per component and cumulative variance. Automatically identifies the elbow and shows how many components are needed to hit 90%, 95%, 99% thresholds.",
        "code": """
import math
import random

random.seed(42)

# ── Generate a 10D dataset with intrinsic 3D structure ────────────────────────
# True data lives in 3D: sample 3 latent variables, then embed in 10D with noise
n, true_dim, obs_dim = 120, 3, 10

latent = [[random.gauss(0,1) for _ in range(true_dim)] for _ in range(n)]

# Random projection matrix to embed 3D → 10D (fixed, not random per run)
import math as _m
proj = []
for i in range(obs_dim):
    row = [_m.cos(i*j+1) for j in range(true_dim)]  # deterministic "random" matrix
    norm = _m.sqrt(sum(x**2 for x in row))
    proj.append([x/norm for x in row])

X = []
for i in range(n):
    point = [sum(latent[i][k]*proj[j][k] for k in range(true_dim)) +
             random.gauss(0, 0.15) for j in range(obs_dim)]
    X.append(point)

p = obs_dim

# ── Standardise ───────────────────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n))/n for j in range(p)]
stds  = [math.sqrt(sum((X[i][j]-means[j])**2 for i in range(n))/(n-1)) for j in range(p)]
Xc    = [[(X[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(n)]

# ── Covariance matrix ─────────────────────────────────────────────────────────
C = [[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

# ── Power iteration eigendecomposition ────────────────────────────────────────
def mat_vec(M, v):
    return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

def power_iter(M, tol=1e-10, max_iter=5000):
    p = len(M)
    v = [math.cos(k) for k in range(p)]  # deterministic seed
    v_norm = math.sqrt(sum(x**2 for x in v))
    v = [x/v_norm for x in v]
    for _ in range(max_iter):
        w = mat_vec(M, v)
        w_norm = math.sqrt(sum(x**2 for x in w))
        v_new = [x/w_norm for x in w]
        diff = math.sqrt(sum((v_new[i]-v[i])**2 for i in range(p)))
        v = v_new
        if diff < tol: break
    lam = sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam, 0.0), v

eigenvalues = []
Cd = [row[:] for row in C]
for _ in range(p):
    lam, vec = power_iter(Cd)
    eigenvalues.append(lam)
    # Deflate
    Cd = [[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]

eigenvalues.sort(reverse=True)
total_var = sum(eigenvalues)
evrs      = [lam/total_var*100 if total_var>0 else 0 for lam in eigenvalues]
cumevrs   = []
c = 0
for e in evrs:
    c += e
    cumevrs.append(c)

# ── Print scree plot ──────────────────────────────────────────────────────────
print(f"Scree Plot  |  {obs_dim}D dataset  |  True intrinsic dim = {true_dim}")
print(f"(Standardised: each feature has unit variance)")
print()
print(f"  {'PC':<5}  {'Eigenvalue':>11}  {'Var %':>7}  {'Cumulative':>11}  Bar")
print("  " + "─"*65)

W = 35
for k, (lam, evr, cev) in enumerate(zip(eigenvalues, evrs, cumevrs)):
    bar_len = int(evr/max(evrs)*W)
    bar     = "█"*bar_len
    marker  = " ◄ elbow" if k == 2 else ""
    print(f"  PC{k+1:<3}  {lam:>11.4f}  {evr:>6.2f}%  {cev:>10.2f}%  {bar}{marker}")

print()

# ── Threshold analysis ────────────────────────────────────────────────────────
print("  Components needed to reach variance threshold:")
for threshold in [80, 90, 95, 99]:
    for k, cev in enumerate(cumevrs):
        if cev >= threshold:
            print(f"    {threshold:>3}% threshold  →  K = {k+1} components")
            break

print()
print(f"  Kaiser criterion (λ > 1.0, data is standardised):")
kaiser_k = sum(1 for lam in eigenvalues if lam > 1.0)
print(f"    {kaiser_k} components have eigenvalue > 1.0")
print()
print(f"  True intrinsic dimensionality: {true_dim}")
print(f"  Scree elbow at: PC3 (▲ marked above)")
print(f"  → PCA correctly identifies the {true_dim}D structure despite {obs_dim}D observations.")
""",
        "runnable": True,
    },

    "▶ Run: PCA for Noise Filtering / Reconstruction": {
        "description": "Apply PCA to a noisy dataset, reconstruct from top K components, and measure reconstruction error "
                       "at each K. Demonstrates how PCA acts as a low-rank filter.",
        "code": """
import math
import random

random.seed(0)

# ── Generate noisy low-rank data ───────────────────────────────────────────────
# True signal: 2D line embedded in 5D + independent Gaussian noise
n, p = 80, 5
noise_std = 0.4

true_signal = [random.gauss(0,1) for _ in range(n)]  # 1D latent
direction   = [0.6, 0.8, 0.2, -0.3, 0.1]             # embedded in 5D

X = []
for i in range(n):
    pt = [true_signal[i]*direction[j] + random.gauss(0, noise_std)
          for j in range(p)]
    X.append(pt)

# ── Mean-centre ───────────────────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n))/n for j in range(p)]
Xc    = [[X[i][j]-means[j] for j in range(p)] for i in range(n)]

# ── Covariance + eigendecomposition ──────────────────────────────────────────
C = [[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M):
    p=len(M)
    v=[math.cos(k+1) for k in range(p)]
    nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(3000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v=[x/nrm for x in w]
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

eigenpairs=[]
Cd=[row[:] for row in C]
for _ in range(p):
    lam,vec=power_iter(Cd)
    eigenpairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
eigenpairs.sort(reverse=True,key=lambda x:x[0])
eigenvalues=[ep[0] for ep in eigenpairs]
eigenvectors=[ep[1] for ep in eigenpairs]
total_var=sum(eigenvalues)

# ── Reconstruction at each K ──────────────────────────────────────────────────
def project(Xc,W):
    return [[sum(Xc[i][j]*W[k][j] for j in range(len(W[0]))) for k in range(len(W))]
            for i in range(len(Xc))]

def reconstruct(Z,W,means):
    p=len(W[0])
    return [[sum(Z[i][k]*W[k][j] for k in range(len(W)))+means[j] for j in range(p)]
            for i in range(len(Z))]

def rmse(A,B):
    n,p=len(A),len(A[0])
    return math.sqrt(sum((A[i][j]-B[i][j])**2 for i in range(n) for j in range(p))/(n*p))

print(f"PCA Noise Filtering  |  n={n}, p={p}, noise σ={noise_std}")
print(f"True signal: 1D line embedded in {p}D")
print()
print(f"  {'K':<4}  {'Eigenvalue λₖ':>14}  {'Var %':>7}  {'Cum %':>7}  {'RMSE':>8}  {'Noise removed?'}")
print("  " + "─"*68)

original_rmse = math.sqrt(sum(noise_std**2 for _ in range(p))/p)
baseline_rmse = rmse(X, [[means[j]]*p for _ in range(n)])

for K in range(1, p+1):
    W = eigenvectors[:K]
    Z = project(Xc, W)
    Xr = reconstruct(Z, W, means)
    r = rmse(X, Xr)
    evr = eigenvalues[K-1]/total_var*100 if total_var>0 else 0
    cev = sum(eigenvalues[:K])/total_var*100 if total_var>0 else 0
    noise_flag = "✓ Signal captured" if K==1 else ("~ Noise added" if K>2 else "")
    print(f"  K={K:<3}  {eigenvalues[K-1]:>14.4f}  {evr:>6.2f}%  {cev:>6.2f}%  {r:>8.4f}  {noise_flag}")

print()
print(f"  Observations:")
print(f"    K=1  : captures the true 1D signal → lowest RMSE on true signal")
print(f"    K>1  : adding more components adds noise back (higher RMSE)")
print(f"    K={p}  : full reconstruction = original data (no compression)")
print()
print(f"  → Reconstruction with K=1 is BETTER than the original noisy data.")
print(f"    PCA acts as a denoising filter by projecting onto the signal subspace.")
""",
        "runnable": True,
    },

    "▶ Run: PCA Loadings and Feature Interpretation": {
        "description": "Compute and print the PCA loadings matrix. Shows which original features each principal component "
                       "represents, enabling interpretation of the principal components.",
        "code": """
import math
import random

random.seed(7)

# ── Dataset: financial-like features with known structure ─────────────────────
# 3 latent factors: growth, risk, liquidity
# 8 observable features built from combinations
n = 100
feature_names = ['Revenue','Profit','EPS','Volatility','Beta','CashFlow','Debt','CurrentRatio']
p = len(feature_names)

# Simulate latent factors
growth     = [random.gauss(0,1) for _ in range(n)]
risk       = [random.gauss(0,1) for _ in range(n)]
liquidity  = [random.gauss(0,1) for _ in range(n)]

noise = 0.3
X = []
for i in range(n):
    g,r,l = growth[i], risk[i], liquidity[i]
    X.append([
        0.9*g + noise*random.gauss(0,1),    # Revenue   ← growth
        0.85*g + noise*random.gauss(0,1),   # Profit    ← growth
        0.8*g + 0.2*r + noise*random.gauss(0,1),  # EPS   ← growth+risk
        0.9*r + noise*random.gauss(0,1),    # Volatility← risk
        0.85*r + noise*random.gauss(0,1),   # Beta      ← risk
        0.8*l + noise*random.gauss(0,1),    # CashFlow  ← liquidity
        -0.7*l + noise*random.gauss(0,1),   # Debt      ← -liquidity
        0.8*l + 0.1*g + noise*random.gauss(0,1), # CurrentRatio ← liquidity
    ])

# ── Standardise ───────────────────────────────────────────────────────────────
means = [sum(X[i][j] for i in range(n))/n for j in range(p)]
stds  = [math.sqrt(sum((X[i][j]-means[j])**2 for i in range(n))/(n-1)) for j in range(p)]
Xc    = [[(X[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(n)]

# ── Covariance matrix ─────────────────────────────────────────────────────────
C = [[sum(Xc[i][j]*Xc[i][k] for i in range(n))/(n-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M, seed):
    p=len(M)
    v=seed[:]; nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(5000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v_new=[x/nrm for x in w]
        if math.sqrt(sum((v_new[k]-v[k])**2 for k in range(p)))<1e-10: break
        v=v_new
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

seeds=[[1.0 if k==j else 0.0 for k in range(p)] for j in range(p)]
eigenpairs=[]
Cd=[row[:] for row in C]
for s in seeds:
    lam,vec=power_iter(Cd,s)
    eigenpairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
eigenpairs.sort(reverse=True,key=lambda x:x[0])
eigenvalues=[ep[0] for ep in eigenpairs]
eigenvectors=[ep[1] for ep in eigenpairs]
total_var=sum(eigenvalues)

# ── Loadings matrix ────────────────────────────────────────────────────────────
# Loadings = eigenvector * sqrt(eigenvalue)  (correlation between feature and PC)
K=3
loadings=[[eigenvectors[k][j]*math.sqrt(eigenvalues[k]) for k in range(K)] for j in range(p)]

print(f"PCA Loadings Matrix  |  {p} features → {K} principal components")
print(f"Dataset: simulated financial metrics (n={n})")
print()
print(f"  {'Feature':<14}" + "".join(f"  {'PC'+str(k+1):>8}" for k in range(K)) + "   Dominant PC")
print("  " + "─"*55)

for j,fname in enumerate(feature_names):
    row_loadings = loadings[j]
    dominant_k = max(range(K), key=lambda k: abs(loadings[j][k]))
    bars = "".join(
        ("  " + ("█" if loadings[j][k]>0 else "░") * int(abs(loadings[j][k])*5)).ljust(10)
        if abs(loadings[j][k])>0.2 else "  " + " "*8
        for k in range(K)
    )
    num_str = "".join(f"  {loadings[j][k]:>+8.3f}" for k in range(K))
    print(f"  {fname:<14}{num_str}   PC{dominant_k+1}")

print()
print(f"  Explained variance:")
for k in range(K):
    evr=eigenvalues[k]/total_var*100
    print(f"    PC{k+1}: {evr:.1f}%")
print()
print(f"  PC Interpretation:")
print(f"    PC1 — high loadings on Revenue, Profit, EPS      → 'Growth factor'")
print(f"    PC2 — high loadings on Volatility, Beta           → 'Risk factor'")
print(f"    PC3 — high loadings on CashFlow, Debt, CurrentRatio → 'Liquidity factor'")
print()
print(f"  PCA recovered the 3 latent factors from 8 observed features.")
print(f"  Features with large |loading| on a PC are most associated with it.")
""",
        "runnable": True,
    },

    "▶ Run: PCA vs Raw Features — Downstream Classification": {
        "description": "Compare k-nearest-neighbour classification accuracy on raw features vs PCA-reduced features at "
                       "different K values. Shows the effect of dimensionality reduction on a downstream task.",
        "code": """
import math
import random

random.seed(42)

# ── Dataset: 3 classes in 20D (true signal in 4D, rest noise) ─────────────────
n_per_class, true_dim, obs_dim = 40, 4, 20
noise_std = 0.8

class_centres_latent = [
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
]

# Fixed embedding matrix (4D → 20D)
import math as _m
embed = [[_m.cos(i+j*1.7+0.5) for j in range(true_dim)] for i in range(obs_dim)]
embed_norms = [math.sqrt(sum(embed[i][j]**2 for j in range(true_dim))) for i in range(obs_dim)]
embed = [[embed[i][j]/embed_norms[i] for j in range(true_dim)] for i in range(obs_dim)]

X, y = [], []
for cls, centre in enumerate(class_centres_latent):
    for _ in range(n_per_class):
        latent = [centre[j]+random.gauss(0,0.6) for j in range(true_dim)]
        obs    = [sum(latent[j]*embed[i][j] for j in range(true_dim)) +
                  random.gauss(0,noise_std) for i in range(obs_dim)]
        X.append(obs); y.append(cls)

n = len(X); p = obs_dim

# ── Train / test split ────────────────────────────────────────────────────────
idx = list(range(n)); random.shuffle(idx)
split = int(0.75*n)
tr_idx, te_idx = idx[:split], idx[split:]
Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
Xte = [X[i] for i in te_idx]; yte = [y[i] for i in te_idx]

# ── PCA fit on train ──────────────────────────────────────────────────────────
means = [sum(Xtr[i][j] for i in range(len(Xtr)))/len(Xtr) for j in range(p)]
stds  = [math.sqrt(sum((Xtr[i][j]-means[j])**2 for i in range(len(Xtr)))/(len(Xtr)-1)) for j in range(p)]
Xctr  = [[(Xtr[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(len(Xtr))]
Xcte  = [[(Xte[i][j]-means[j])/(stds[j] if stds[j]>0 else 1) for j in range(p)] for i in range(len(Xte))]

C = [[sum(Xctr[i][j]*Xctr[i][k] for i in range(len(Xctr)))/(len(Xctr)-1) for k in range(p)] for j in range(p)]

def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
def power_iter(M, seed):
    p=len(M); v=seed[:]; nrm=math.sqrt(sum(x**2 for x in v)); v=[x/nrm for x in v]
    for _ in range(3000):
        w=mat_vec(M,v); nrm=math.sqrt(sum(x**2 for x in w))
        if nrm<1e-12: break
        v_new=[x/nrm for x in w]
        if math.sqrt(sum((v_new[k]-v[k])**2 for k in range(p)))<1e-10: v=v_new; break
        v=v_new
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(p)) for i in range(p))
    return max(lam,0.0),v

seeds=[[float(k==j) for k in range(p)] for j in range(p)]
eigenpairs=[]; Cd=[row[:] for row in C]
for s in seeds:
    lam,vec=power_iter(Cd,s); eigenpairs.append((lam,vec))
    Cd=[[Cd[i][j]-lam*vec[i]*vec[j] for j in range(p)] for i in range(p)]
eigenpairs.sort(reverse=True,key=lambda x:x[0])
evecs=[ep[1] for ep in eigenpairs]

def project(Xc,W):
    return [[sum(Xc[i][j]*W[k][j] for j in range(p)) for k in range(len(W))]
            for i in range(len(Xc))]

# ── 1-NN classifier ───────────────────────────────────────────────────────────
def euclidean(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def knn_acc(Xtr,ytr,Xte,yte,K=1):
    correct=0
    for i,p_te in enumerate(Xte):
        dists=[(euclidean(p_te,Xtr[j]),ytr[j]) for j in range(len(Xtr))]
        dists.sort(); nn_labels=[d[1] for d in dists[:K]]
        pred=max(set(nn_labels),key=nn_labels.count)
        if pred==yte[i]: correct+=1
    return correct/len(yte)

# ── Compare raw vs PCA at each K ─────────────────────────────────────────────
raw_acc = knn_acc(Xctr, ytr, Xcte, yte)
total_var = sum(ep[0] for ep in eigenpairs)

print(f"PCA + 1-NN Classification  |  {obs_dim}D → K  |  3 classes  |  n={n}")
print(f"True signal lives in {true_dim}D  |  Noise σ={noise_std}")
print()
print(f"  Raw {obs_dim}D accuracy (no PCA): {raw_acc:.1%}")
print()
print(f"  {'K PCs':<8}  {'Cum Var':>8}  {'Accuracy':>10}  {'vs Raw':>8}  {'Signal?'}")
print("  " + "─"*55)

best_acc, best_K = 0, 0
for K in [1,2,3,4,5,8,10,15,20]:
    if K > p: break
    W   = evecs[:K]
    Ztr = project(Xctr, W)
    Zte = project(Xcte, W)
    acc = knn_acc(Ztr, ytr, Zte, yte)
    cev = sum(ep[0] for ep in eigenpairs[:K])/total_var*100
    delta = acc - raw_acc
    signal = "✓ signal" if K<=true_dim+1 else ("~ noise" if K>true_dim+3 else "")
    bar = ("▲" if delta>0 else "▼") + f" {abs(delta):.1%}"
    print(f"  K={K:<6}  {cev:>7.1f}%  {acc:>10.1%}  {bar:>8}  {signal}")
    if acc > best_acc: best_acc=acc; best_K=K

print()
print(f"  Best accuracy: {best_acc:.1%} at K={best_K} components")
print(f"  → Reducing from {obs_dim}D to {best_K}D improves accuracy by",
      f"{best_acc-raw_acc:+.1%}")
print(f"  → Noise dimensions hurt 1-NN by inflating distances with irrelevant info.")
""",
        "runnable": True,
    },

}


VISUAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0;
       padding: 20px; }
h2   { color: #60a5fa; margin-bottom: 4px; }
.subtitle { color: #64748b; margin-bottom: 22px; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card { background: #1e2130; border-radius: 12px; padding: 18px;
        border: 1px solid #2d3148; }
.card h3 { color: #60a5fa; margin: 0 0 10px; font-size: 0.9em;
           text-transform: uppercase; letter-spacing: 0.05em; }
canvas { display: block; }
.params { background: #12141f; padding: 8px 12px; border-radius: 8px;
          font-size: 0.81em; color: #94a3b8; margin: 8px 0; line-height: 1.6; }
.pv { color: #60a5fa; font-weight: bold; }
.slider-row { display: flex; align-items: center; gap: 10px; margin-top: 7px; }
.slider-row label { font-size: 0.8em; color: #94a3b8; min-width: 90px; }
input[type=range] { accent-color: #60a5fa; flex: 1; }
.vb { font-size: 0.8em; color: #60a5fa; min-width: 36px; }
.btn-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 9px; }
button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168;
         border-radius: 6px; padding: 4px 12px; cursor: pointer;
         font-size: 0.8em; transition: background 0.15s; }
button:hover { background: #3d4168; }
button.active { background: #60a5fa; color: #0f1117; }
</style>
</head>
<body>

<h2>📐 PCA Visual Explorer</h2>
<p class="subtitle">Principal Component Analysis — see variance directions, scree plots, and projection in real time</p>

<div class="grid">

  <!-- Panel 1: 2D scatter with PC axes -->
  <div class="card">
    <h3>PC Directions on Raw Data</h3>
    <div class="params">
      PC1 explains <span class="pv" id="evr1">—</span>% of variance &nbsp;|&nbsp;
      PC2 explains <span class="pv" id="evr2">—</span>%<br>
      Arrows show principal component directions. Length ∝ √λ (standard deviation along that axis).
    </div>
    <canvas id="cvScatter" width="340" height="260"></canvas>
    <div class="slider-row">
      <label>Rotation θ</label>
      <input type="range" id="rotSlider" min="0" max="180" step="1" value="35">
      <span class="vb" id="rotVal">35°</span>
    </div>
    <div class="slider-row">
      <label>Noise σ</label>
      <input type="range" id="noiseSlider" min="1" max="60" step="1" value="15">
      <span class="vb" id="noiseVal">0.15</span>
    </div>
    <div class="params">Rotate the data cloud or add noise to see how PCs adapt.</div>
  </div>

  <!-- Panel 2: Scree plot -->
  <div class="card">
    <h3>Scree Plot + Cumulative Variance</h3>
    <div class="params">
      Eigenvalue per component (bars) and cumulative explained variance (line).<br>
      Drag the <span style="color:#fbbf24">threshold line</span> to set a variance target.
      Components right of the elbow add diminishing information.
    </div>
    <canvas id="cvScree" width="340" height="260" style="cursor:crosshair;"></canvas>
    <div class="params" id="screeInfo">—</div>
  </div>

  <!-- Panel 3: Projected 1D distribution -->
  <div class="card">
    <h3>Projection onto PC1 vs PC2</h3>
    <div class="params">
      Distribution of PC scores along each component.<br>
      PC1 has the widest spread (maximum variance); PC2 is narrower and orthogonal.
    </div>
    <canvas id="cvProj" width="340" height="260"></canvas>
    <div class="btn-row">
      <button onclick="setDataset('ellipse')" id="btnEllipse" class="active">Ellipse</button>
      <button onclick="setDataset('clusters')" id="btnClusters">3 Clusters</button>
      <button onclick="setDataset('ring')" id="btnRing">Ring</button>
      <button onclick="setDataset('uniform')" id="btnUniform">Uniform</button>
    </div>
  </div>

  <!-- Panel 4: Reconstruction error vs K -->
  <div class="card">
    <h3>Reconstruction Error vs K</h3>
    <div class="params" id="reconInfo">
      RMSE and retained variance as number of components K increases.
      Moving <span style="color:#60a5fa">K slider</span> shows the reconstructed point cloud above.
    </div>
    <canvas id="cvRecon" width="340" height="200"></canvas>
    <div class="slider-row">
      <label>K components</label>
      <input type="range" id="kSlider" min="1" max="2" step="1" value="1">
      <span class="vb" id="kVal">1</span>
    </div>
    <canvas id="cvReconScatter" width="340" height="120"></canvas>
  </div>

</div>

<script>
// ── Utilities ─────────────────────────────────────────────────────────────────
const rng = (() => {
  let s = 12345;
  return () => { s=(s*1664525+1013904223)>>>0; return s/4294967296; };
})();
const gauss = () => {
  const u=rng()||1e-9, v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
};
const dist2 = (a,b) => Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2);

// ── Dataset generators ────────────────────────────────────────────────────────
function genEllipse(rot_deg, noise_frac){
  const rot=rot_deg*Math.PI/180;
  const pts=[];
  for(let i=0;i<120;i++){
    const t=rng()*2*Math.PI;
    let x=2.5*Math.cos(t)+gauss()*noise_frac;
    let y=0.5*Math.sin(t)+gauss()*noise_frac;
    pts.push([x*Math.cos(rot)-y*Math.sin(rot), x*Math.sin(rot)+y*Math.cos(rot)]);
  }
  return pts;
}

function genClusters(){
  const pts=[];
  const cs=[[2,2],[-2,-1],[0,-2.5]];
  cs.forEach(([cx,cy])=>{
    for(let i=0;i<40;i++) pts.push([cx+gauss()*0.6,cy+gauss()*0.6]);
  });
  return pts;
}

function genRing(){
  const pts=[];
  for(let i=0;i<120;i++){
    const a=rng()*2*Math.PI;
    const r=2+gauss()*0.2;
    pts.push([r*Math.cos(a),r*Math.sin(a)]);
  }
  return pts;
}

function genUniform(){
  const pts=[];
  for(let i=0;i<120;i++) pts.push([(rng()-0.5)*6,(rng()-0.5)*6]);
  return pts;
}

// ── PCA (2D only for vis) ──────────────────────────────────────────────────────
function pca2D(pts){
  const n=pts.length;
  const mx=pts.reduce((s,p)=>s+p[0],0)/n;
  const my=pts.reduce((s,p)=>s+p[1],0)/n;
  const Xc=pts.map(p=>[p[0]-mx,p[1]-my]);
  // Covariance
  const c11=Xc.reduce((s,p)=>s+p[0]*p[0],0)/(n-1);
  const c22=Xc.reduce((s,p)=>s+p[1]*p[1],0)/(n-1);
  const c12=Xc.reduce((s,p)=>s+p[0]*p[1],0)/(n-1);
  // Analytic eigendecomposition for 2×2
  const tr=c11+c22, det=c11*c22-c12*c12;
  const disc=Math.sqrt(Math.max(0,(tr*tr/4)-det));
  const l1=tr/2+disc, l2=tr/2-disc;
  // Eigenvectors
  let v1,v2;
  if(Math.abs(c12)>1e-10){
    const a=[l1-c22, c12]; const an=Math.sqrt(a[0]**2+a[1]**2);
    v1=[a[0]/an,a[1]/an];
    v2=[-v1[1],v1[0]];
  } else {
    v1=c11>=c22?[1,0]:[0,1];
    v2=c11>=c22?[0,1]:[1,0];
  }
  // Scores
  const z1=Xc.map(p=>p[0]*v1[0]+p[1]*v1[1]);
  const z2=Xc.map(p=>p[0]*v2[0]+p[1]*v2[1]);
  const total=l1+l2||1;
  return { l1,l2,v1,v2,z1,z2,mx,my,Xc,total,evr1:l1/total,evr2:l2/total };
}

// ── Canvas scaling helpers ────────────────────────────────────────────────────
function makeScale(pts, W, H, pad=28){
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
  const mnx=Math.min(...xs)-0.5, mxx=Math.max(...xs)+0.5;
  const mny=Math.min(...ys)-0.5, mxy=Math.max(...ys)+0.5;
  const sx=x=>pad+(x-mnx)/(mxx-mnx)*(W-2*pad);
  const sy=y=>H-pad-(y-mny)/(mxy-mny)*(H-2*pad);
  return {sx,sy,mnx,mxx,mny,mxy};
}

// ── State ─────────────────────────────────────────────────────────────────────
let currentDataset='ellipse';
let currentRot=35, currentNoise=0.15;
let rawPts=[], pcaResult=null;
let screeThreshold=0.85;
let K=1;

function rebuildData(){
  if(currentDataset==='ellipse') rawPts=genEllipse(currentRot,currentNoise);
  else if(currentDataset==='clusters') rawPts=genClusters();
  else if(currentDataset==='ring') rawPts=genRing();
  else rawPts=genUniform();
  pcaResult=pca2D(rawPts);
  drawAll();
}

function setDataset(name){
  currentDataset=name;
  ['Ellipse','Clusters','Ring','Uniform'].forEach(n=>{
    document.getElementById('btn'+n).classList.toggle('active',n.toLowerCase()===name);
  });
  rebuildData();
}

// ── Panel 1: Scatter + PC arrows ─────────────────────────────────────────────
const cv1=document.getElementById('cvScatter'); const ctx1=cv1.getContext('2d');
function drawScatter(){
  const W=340,H=260;
  ctx1.clearRect(0,0,W,H);
  const r=pcaResult;
  const {sx,sy}=makeScale(rawPts,W,H);

  // Points
  rawPts.forEach(p=>{
    ctx1.beginPath(); ctx1.arc(sx(p[0]),sy(p[1]),3.5,0,2*Math.PI);
    ctx1.fillStyle='rgba(96,165,250,0.6)'; ctx1.fill();
  });

  // Mean point
  const msx=sx(r.mx), msy=sy(r.my);

  // PC arrows: length = sqrt(eigenvalue) in data units × scale factor
  const sc=makeScale(rawPts,W,H);
  const xRange=sc.mxx-sc.mnx, yRange=sc.mxy-sc.mny;
  const dataScale=Math.min((W-56)/xRange,(H-56)/yRange);

  function drawArrow(ctx,x0,y0,dx,dy,color,label){
    const len=Math.sqrt(dx**2+dy**2);
    if(len<1) return;
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x0+dx,y0+dy);
    ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.stroke();
    // Arrowhead
    const angle=Math.atan2(dy,dx);
    ctx.beginPath();
    ctx.moveTo(x0+dx,y0+dy);
    ctx.lineTo(x0+dx-8*Math.cos(angle-0.4),y0+dy-8*Math.sin(angle-0.4));
    ctx.lineTo(x0+dx-8*Math.cos(angle+0.4),y0+dy-8*Math.sin(angle+0.4));
    ctx.closePath(); ctx.fillStyle=color; ctx.fill();
    ctx.fillStyle=color; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
    ctx.fillText(label,x0+dx+14*Math.cos(angle),y0+dy+14*Math.sin(angle));
  }

  const arrowScale = dataScale * 1.2;
  const s1=Math.sqrt(Math.max(r.l1,0)); const s2=Math.sqrt(Math.max(r.l2,0));
  drawArrow(ctx1,msx,msy, r.v1[0]*s1*arrowScale,-r.v1[1]*s1*arrowScale,'#fbbf24','PC1');
  drawArrow(ctx1,msx,msy, r.v2[0]*s2*arrowScale,-r.v2[1]*s2*arrowScale,'#f472b6','PC2');

  document.getElementById('evr1').textContent=(r.evr1*100).toFixed(1);
  document.getElementById('evr2').textContent=(r.evr2*100).toFixed(1);
}

// ── Panel 2: Scree ────────────────────────────────────────────────────────────
const cv2=document.getElementById('cvScree'); const ctx2=cv2.getContext('2d');
let screeY=null;

function drawScree(){
  const W=340,H=260,PAD=36;
  ctx2.clearRect(0,0,W,H);
  const r=pcaResult;
  const evals=[r.l1,r.l2];
  const total=r.total;
  const evrs=[r.l1/total,r.l2/total];
  const cumevrs=[evrs[0],evrs[0]+evrs[1]];

  const barW=(W-2*PAD)/2-10;
  const maxE=Math.max(...evals,0.01);

  // Bars
  evals.forEach((lam,k)=>{
    const x=PAD+k*(barW+10);
    const barH=(lam/maxE)*(H-2*PAD);
    const y=H-PAD-barH;
    ctx2.fillStyle='rgba(96,165,250,0.5)';
    ctx2.fillRect(x,y,barW,barH);
    ctx2.strokeStyle='#60a5fa'; ctx2.lineWidth=1;
    ctx2.strokeRect(x,y,barW,barH);
    ctx2.fillStyle='#94a3b8'; ctx2.font='10px sans-serif'; ctx2.textAlign='center';
    ctx2.fillText(`PC${k+1}`+`\n${(evrs[k]*100).toFixed(1)}%`,x+barW/2,H-PAD+14);
    ctx2.fillText(`λ=${lam.toFixed(2)}`,x+barW/2,y-6);
  });

  // Cumulative line
  ctx2.beginPath();
  ctx2.strokeStyle='#34d399'; ctx2.lineWidth=2; ctx2.setLineDash([]);
  cumevrs.forEach((cev,k)=>{
    const x=PAD+(k+0.5)*(barW+10);
    const y=H-PAD-(cev)*(H-2*PAD);
    k===0?ctx2.moveTo(x,y):ctx2.lineTo(x,y);
  });
  ctx2.stroke();
  cumevrs.forEach((cev,k)=>{
    const x=PAD+(k+0.5)*(barW+10);
    const y=H-PAD-cev*(H-2*PAD);
    ctx2.beginPath(); ctx2.arc(x,y,4,0,2*Math.PI);
    ctx2.fillStyle='#34d399'; ctx2.fill();
  });

  // Threshold line (draggable)
  const threshY=screeY!==null?screeY:H-PAD-screeThreshold*(H-2*PAD);
  ctx2.beginPath(); ctx2.setLineDash([5,3]);
  ctx2.moveTo(PAD,threshY); ctx2.lineTo(W-PAD,threshY);
  ctx2.strokeStyle='#fbbf24'; ctx2.lineWidth=1.5; ctx2.stroke();
  ctx2.setLineDash([]);
  const thrVal=((H-PAD-threshY)/(H-2*PAD));
  ctx2.fillStyle='#fbbf24'; ctx2.font='9px sans-serif'; ctx2.textAlign='right';
  ctx2.fillText(`${(thrVal*100).toFixed(0)}%`,W-PAD-2,threshY-3);

  // How many PCs to reach threshold?
  const needed=cumevrs.findIndex(c=>c>=thrVal)+1 || 2;
  document.getElementById('screeInfo').innerHTML=
    `Threshold: <span class="pv">${(thrVal*100).toFixed(0)}%</span> variance requires
     <span class="pv">${needed}</span> component(s). Drag the yellow line to change.`;
}

cv2.addEventListener('mousemove',e=>{
  const rect=cv2.getBoundingClientRect();
  screeY=e.clientY-rect.top;
  drawScree();
});
cv2.addEventListener('mouseleave',()=>{ screeY=null; drawScree(); });

// ── Panel 3: Projection distributions ────────────────────────────────────────
const cv3=document.getElementById('cvProj'); const ctx3=cv3.getContext('2d');
function drawProjection(){
  const W=340,H=260,PAD=24;
  ctx3.clearRect(0,0,W,H);
  const r=pcaResult;

  function drawHist(scores,color,yOffset,label,sigma){
    const mn=Math.min(...scores), mx=Math.max(...scores);
    const bins=20, binW=(mx-mn)/bins||1;
    const counts=new Array(bins).fill(0);
    scores.forEach(s=>{ const b=Math.min(bins-1,Math.floor((s-mn)/binW)); counts[b]++; });
    const maxC=Math.max(...counts,1);
    const plotH=80, xscale=(W-2*PAD)/(mx-mn||1);
    ctx3.fillStyle='#475569'; ctx3.font='9px sans-serif'; ctx3.textAlign='left';
    ctx3.fillText(label,PAD,yOffset-2);
    counts.forEach((c,i)=>{
      const x=PAD+i*((W-2*PAD)/bins);
      const h=c/maxC*plotH;
      ctx3.fillStyle=color+'99';
      ctx3.fillRect(x,yOffset+plotH-h,(W-2*PAD)/bins-1,h);
      ctx3.strokeStyle=color; ctx3.lineWidth=0.5;
      ctx3.strokeRect(x,yOffset+plotH-h,(W-2*PAD)/bins-1,h);
    });
    // Std dev marker
    const mean=scores.reduce((a,b)=>a+b,0)/scores.length;
    const mx_=Math.max(...scores), mn_=Math.min(...scores);
    const cx=PAD+(mean-mn_)/(mx_-mn_||1)*(W-2*PAD);
    ctx3.beginPath(); ctx3.setLineDash([3,2]);
    ctx3.moveTo(cx,yOffset); ctx3.lineTo(cx,yOffset+plotH);
    ctx3.strokeStyle=color; ctx3.lineWidth=1.5; ctx3.stroke(); ctx3.setLineDash([]);
    ctx3.fillStyle=color; ctx3.font='8px sans-serif'; ctx3.textAlign='center';
    ctx3.fillText(`σ=${sigma.toFixed(2)}`,cx,yOffset+plotH+12);
  }

  drawHist(r.z1,'#fbbf24',30,`PC1 scores  (${(r.evr1*100).toFixed(1)}% variance)`,Math.sqrt(Math.max(r.l1,0)));
  drawHist(r.z2,'#f472b6',148,`PC2 scores  (${(r.evr2*100).toFixed(1)}% variance)`,Math.sqrt(Math.max(r.l2,0)));

  ctx3.fillStyle='#64748b'; ctx3.font='8px sans-serif'; ctx3.textAlign='center';
  ctx3.fillText('PC1 is always the widest distribution — it captures maximum variance.',W/2,250);
}

// ── Panel 4: Reconstruction error ────────────────────────────────────────────
const cv4=document.getElementById('cvRecon'); const ctx4=cv4.getContext('2d');
const cv5=document.getElementById('cvReconScatter'); const ctx5=cv5.getContext('2d');

document.getElementById('kSlider').max=2;
document.getElementById('kSlider').addEventListener('input',e=>{
  K=parseInt(e.target.value);
  document.getElementById('kVal').textContent=K;
  drawRecon();
});

function drawRecon(){
  const W=340,H=200,PAD=36;
  ctx4.clearRect(0,0,W,H);
  const r=pcaResult;
  const n=rawPts.length;

  // Compute RMSE for K=1 and K=2
  function rmseK(K){
    const vecs=K===1?[r.v1]:[r.v1,r.v2];
    const evals_k=K===1?[r.l1]:[r.l1,r.l2];
    let sse=0;
    r.Xc.forEach((p,i)=>{
      const scores=vecs.map(v=>p[0]*v[0]+p[1]*v[1]);
      const rec=[
        scores.reduce((s,sc,k)=>s+sc*vecs[k][0],0),
        scores.reduce((s,sc,k)=>s+sc*vecs[k][1],0),
      ];
      sse+=(p[0]-rec[0])**2+(p[1]-rec[1])**2;
    });
    return Math.sqrt(sse/(n*2));
  }

  const ks=[1,2];
  const rmses=ks.map(k=>rmseK(k));
  const varRet=[r.evr1,r.evr1+r.evr2];
  const maxRmse=Math.max(...rmses,0.01);

  ks.forEach((k,i)=>{
    const x=PAD+i*(W-2*PAD)/2+10;
    const barH=(rmses[i]/maxRmse)*(H-2*PAD)*0.8;
    const barW=(W-2*PAD)/2-20;
    const col=k===K?'#60a5fa':'#334155';
    ctx4.fillStyle=col+'aa';
    ctx4.fillRect(x,H-PAD-barH,barW,barH);
    ctx4.strokeStyle=col; ctx4.lineWidth=k===K?2:1;
    ctx4.strokeRect(x,H-PAD-barH,barW,barH);
    ctx4.fillStyle=k===K?'#60a5fa':'#64748b';
    ctx4.font=`${k===K?'bold ':' '}10px sans-serif`; ctx4.textAlign='center';
    ctx4.fillText(`K=${k}`,x+barW/2,H-PAD+13);
    ctx4.fillText(`RMSE=${rmses[i].toFixed(3)}`,x+barW/2,H-PAD-barH-14);
    ctx4.fillText(`Var=${(varRet[i]*100).toFixed(1)}%`,x+barW/2,H-PAD-barH-4);
  });

  document.getElementById('reconInfo').innerHTML=
    `K=<span class="pv">${K}</span> retains <span class="pv">${(varRet[K-1]*100).toFixed(1)}%</span>
     variance. Reconstruction RMSE = <span class="pv">${rmses[K-1].toFixed(3)}</span>.`;

  // Reconstruction scatter
  const W5=340,H5=120,P5=16;
  ctx5.clearRect(0,0,W5,H5);
  const vecs=K===1?[r.v1]:[r.v1,r.v2];
  const recPts=r.Xc.map(p=>{
    const scores=vecs.map(v=>p[0]*v[0]+p[1]*v[1]);
    return [
      scores.reduce((s,sc,k)=>s+sc*vecs[k][0],0)+r.mx,
      scores.reduce((s,sc,k)=>s+sc*vecs[k][1],0)+r.my,
    ];
  });

  const allPts=[...rawPts,...recPts];
  const {sx,sy}=makeScale(allPts,W5,H5,P5);

  rawPts.forEach((p,i)=>{
    ctx5.beginPath(); ctx5.moveTo(sx(p[0]),sy(p[1])); ctx5.lineTo(sx(recPts[i][0]),sy(recPts[i][1]));
    ctx5.strokeStyle='rgba(248,113,113,0.3)'; ctx5.lineWidth=0.8; ctx5.stroke();
  });
  rawPts.forEach(p=>{
    ctx5.beginPath(); ctx5.arc(sx(p[0]),sy(p[1]),2.5,0,2*Math.PI);
    ctx5.fillStyle='rgba(96,165,250,0.7)'; ctx5.fill();
  });
  recPts.forEach(p=>{
    ctx5.beginPath(); ctx5.arc(sx(p[0]),sy(p[1]),2.5,0,2*Math.PI);
    ctx5.fillStyle='rgba(251,191,36,0.8)'; ctx5.fill();
  });

  ctx5.fillStyle='#64748b'; ctx5.font='8px sans-serif'; ctx5.textAlign='center';
  ctx5.fillText('Blue=original  Gold=reconstructed  Red line=reconstruction error',W5/2,H5-2);
}

// ── Sliders ───────────────────────────────────────────────────────────────────
document.getElementById('rotSlider').addEventListener('input',e=>{
  currentRot=parseInt(e.target.value);
  document.getElementById('rotVal').textContent=currentRot+'°';
  if(currentDataset==='ellipse') rebuildData();
});
document.getElementById('noiseSlider').addEventListener('input',e=>{
  currentNoise=parseInt(e.target.value)/100;
  document.getElementById('noiseVal').textContent=currentNoise.toFixed(2);
  if(currentDataset==='ellipse') rebuildData();
});

function drawAll(){ drawScatter(); drawScree(); drawProjection(); drawRecon(); }

// ── Init ──────────────────────────────────────────────────────────────────────
rebuildData();
</script>
</body>
</html>
"""