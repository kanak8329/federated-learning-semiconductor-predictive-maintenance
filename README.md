🧠 Federated Learning for Predictive Maintenance in Semiconductor Manufacturing
📌 Project Overview

This project investigates Federated Learning (FL) for predictive maintenance in semiconductor manufacturing environments using the SECOM dataset.
The core objective is to design and evaluate privacy-preserving, collaborative LSTM-based models that can predict equipment failures across multiple manufacturing sites (fabs) without sharing raw sensor data.

The project progresses from:

centralized deep learning,

to basic federated learning (FedAvg),

and is designed to be extended toward non-IID data, robust aggregation, and security analysis, reflecting real-world industrial constraints.

🏭 Motivation (Why this matters)

Semiconductor manufacturing systems generate high-dimensional sensor data that is:

sensitive,

proprietary,

and cannot be shared across companies or fabs.

Traditional centralized machine learning:

requires full data sharing,

violates privacy and IP constraints,

is often infeasible in industrial settings.

Federated Learning enables:

collaborative model training,

data privacy preservation,

scalable deployment across distributed manufacturing sites.

This project explores whether federated LSTM models can achieve comparable predictive performance to centralized models while maintaining privacy.

📊 Dataset: SECOM (UCI Machine Learning Repository)

Source: Semiconductor Manufacturing Process Dataset (SECOM)
Samples: 1,567 production instances
Features: 590 sensor measurements
Labels: Binary outcome

1 → Fail

-1 → Pass (converted to 0)

Preprocessing Challenges

Large number of missing values

Separate label file

High dimensionality

Non-time-series structure

Preprocessing Steps Applied

Median imputation for missing values

Feature standardization

Label normalization ({-1, 1} → {0, 1})

Conversion to time-series windows for LSTM modeling

⏱ Time-Series Windowing Strategy

Although SECOM is not originally sequential, manufacturing sensors operate continuously.
To enable LSTM modeling, sliding windows are constructed:

Window length: 10 timesteps

Input shape: (samples, 10, 590)

Label: outcome of the final timestep in each window

This simulates temporal equipment behavior required for predictive maintenance.

🧩 System Architecture
Centralized Learning
SECOM Data → Preprocessing → LSTM Model → Failure Prediction

Federated Learning (FedAvg)
Client 1 (Fab A) ─┐
Client 2 (Fab B) ─┼── FedAvg Aggregation ── Global Model
Client 3 (Fab C) ─┘


Each client trains locally on its own data

Only model weights are shared

Raw sensor data never leaves the client

🧠 Models Used
LSTM-based Binary Classifier

Input: (sequence_length × feature_dim)

Two-layer LSTM

Fully connected classification head

Sigmoid output

Loss: Binary Cross-Entropy

LSTM is chosen due to its ability to capture:

temporal dependencies,

degradation trends,

early failure signals.

🔬 Experimental Pipeline
Step 1 — Data Preprocessing

Merge SECOM sensor and label files

Handle missing values

Normalize features

Save clean dataset

Step 2 — Time-Series Construction

Sliding window generation

LSTM-ready tensors

Train/test split

Step 3 — Centralized Baseline

Train a centralized LSTM model

Establish baseline accuracy and F1-score

Save trained model and learning curves

Step 4 — Federated Learning (FedAvg)

Simulate multiple manufacturing clients

Train local LSTM models independently

Aggregate weights using FedAvg

Evaluate global model after each round

Step 5 — Performance Comparison

Centralized vs Federated evaluation

Accuracy and F1-score comparison

Visualization of trade-offs

📈 Key Results (Typical Observation)

Centralized model achieves slightly higher performance

Federated model achieves comparable accuracy

Performance gap is small considering no data sharing

This validates the feasibility of privacy-preserving collaborative predictive maintenance in semiconductor manufacturing.

🔐 Privacy & Security Perspective

While federated learning improves privacy, it is not inherently secure.

This project is designed to be extended toward:

non-IID client data distributions,

robust aggregation methods,

malicious client simulations,

communication efficiency analysis.

These extensions align closely with real industrial FL challenges.

🛠 Technology Stack

Language: Python

Deep Learning: PyTorch

Machine Learning: Scikit-learn

Federated Learning: Custom FedAvg simulation

Data Processing: NumPy, Pandas

Visualization: Matplotlib

📁 Project Structure
.
├── secom_preprocess.py          # Data cleaning & normalization
├── make_windows.py              # Time-series windowing
├── centralized_train.py         # Centralized LSTM baseline
├── federated_train.py           # Federated learning (FedAvg)
├── compare_models.py            # Performance comparison
├── model.py                     # LSTM architecture
├── utils.py                     # Training & evaluation utilities
├── data/
│   ├── secom.data
│   ├── secom_labels.data
│   ├── secom_clean.csv
│   └── windows/clients/
├── results/
│   ├── centralized_model.pt
│   ├── federated_model.pt
│   └── plots/
└── README.md

🚀 Planned Extensions (Advanced Work)

Non-IID client simulations

Robust aggregation (Trimmed Mean, Median)

Client drift analysis

Malicious client attack simulation

Differential privacy intuition

Communication efficiency experiments

These extensions aim to transform this project into a research-oriented federated learning study.

🎓 Academic Relevance

This project directly aligns with:

Intelligent Manufacturing

Privacy-Preserving AI

Federated Learning

Predictive Maintenance

Semiconductor Equipment Analytics

It is designed as preparation for graduate-level research in AI/ML and smart manufacturing systems.
