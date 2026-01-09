# Star-Classifier
An expansion on the previous project of Star Analytics. Looking into what method can best classify the type of star, while measuring the difference between them and the original set, utilizing both k-fold and LOOCV.

---

## Project Overview

In this AI/ML project I explored and expanded upon the previous project's nuances of Star Type Classification. Through multiple methods such as Logistic Regression, K-NN, Random Forrest Generation, and SVM, I attempt to see which model would best result in the most accurate star classification. Through the process of engineering artificial noise and bootstrapping, I attempted to remedy the overfitting due to the constraints of the dataset. These proved to only highlight further the limitations of the dataset and a thorough glare of this was seen when they were combined.

---

## Dataset

This project uses the **Star Dataset** from Kaggle, which contains physical properties of stars including:

- Temperature (Kelvin)
- Luminosity (relative to the Sun)
- Radius (relative to the Sun)
- Absolute magnitude
- Star type and color

**Dataset link (CC0 / public):** 
You may view the dataset here: [Kaggle Dataset](https://www.kaggle.com/datasets/deepu1109/star-dataset?resource=download).

The dataset is small, clean, and ideal for exploratory analysis and unsupervised learning.

---

# Star Classification with Robust Validation and Scaling Analysis

This project explores multi-class star classification through a **methodology-first lens**, emphasizing validation strategy, robustness testing, and the limitations imposed by dataset construction. Rather than optimizing for maximal performance, the focus is on *understanding model behavior* under realistic constraints.

The work builds on classical statistical intuition and incrementally introduces machine learning techniques, mirroring how such problems are approached in applied research settings.

---

## Project Motivation

Stellar classification is a physically grounded problem where labels are derived from continuous astrophysical properties such as temperature, luminosity, and radius. While machine learning models can achieve strong performance on curated datasets, high accuracy alone does not guarantee meaningful generalization.

This project asks:

> *When models perform extremely well, is that a success of modeling — or a reflection of the dataset itself?*

To answer this, we prioritize:
- principled feature handling
- rigorous cross-validation
- robustness checks
- honest discussion of modeling limits

---

## Methodological Overview

### Feature Representation

- **Continuous physical features** (temperature, luminosity, radius, magnitude) are log-transformed where appropriate to stabilize scale.
- **Categorical astrophysical features** (star color, spectral class) are ordinally encoded using physically motivated orderings rather than naive enumeration.
- Feature scaling is applied uniformly across models.

This preserves interpretability while respecting known stellar structure.

---

### Model Suite

A small but diverse set of baseline models is used:

- Multinomial Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Radial Basis Function SVM

The goal is *comparative behavior*, not hyperparameter tuning.

---

### Validation Strategy (Core Focus)

All evaluation is conducted using **out-of-fold predictions**, avoiding reliance on a single train/test split.

Validation methods include:
- Stratified K-Fold Cross Validation
- Leave-One-Out Cross Validation (LOOCV)

Metrics are reported as:
- mean performance
- variability across folds

This allows stability to be assessed alongside accuracy.

---

### Learning Curve Analysis

Learning curves are used to diagnose:
- bias vs variance behavior
- data sufficiency
- performance saturation

These curves help contextualize high accuracy results and identify when additional data ceases to provide meaningful gains.

---

### Robustness to Measurement Noise

To simulate real-world observational uncertainty, controlled noise is introduced to continuous physical features.

This experiment tests whether models:
- rely on fragile exact values
- or learn stable, physically meaningful relationships

Results show minimal degradation, suggesting strong separability in the feature space.

---

### Synthetic Scaling via Bootstrapping

To investigate sample size limitations, the dataset is expanded to ~1,000 records using **physics-informed bootstrapping**:

- Class proportions are preserved
- Continuous features are sampled within observed distributions
- Mild noise is applied to avoid duplication

This experiment is explicitly framed as a *diagnostic tool*, not a substitute for real data.

Interestingly, performance saturation persists — reinforcing the conclusion that dataset construction and label discretization dominate model behavior.

---

### Model Agreement Analysis

Cross-validated predictions are compared across models to evaluate:
- consensus
- diversity of decision boundaries
- redundancy between approaches

High agreement further supports the interpretation that the task is intrinsically well-separated given the current labels.

---

##  Key Findings

- Strong performance is consistent across models and validation schemes
- Robustness tests and synthetic scaling do not meaningfully degrade results
- Model complexity does not materially affect outcomes
- Dataset structure, rather than model choice, is the primary limiting factor

In short:
> **The problem is solved as posed — and that fact itself is the most important result.**

---

## Limitations and Interpretation

- Star type is a discrete label derived from continuous physics
- Clean class boundaries induce artificial separability
- Synthetic data reinforces existing structure rather than introducing new regimes

These results should be interpreted as a **methodological demonstration**, not a deployment-ready classifier.

---

## Future Directions

Possible extensions include:
- reframing the task as regression on continuous stellar properties
- introducing probabilistic or uncertain labels
- incorporating larger, heterogeneous astrophysical catalogs
