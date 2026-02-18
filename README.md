# Machine Learning-Based Geographic Routing for Sparse Aeronautical Networks

> **🏆 Bachelor Thesis Project** | Grade: 1.0 (Highest possible grade at TUHH)
> 
> **📄 Publication:** Served as the foundational work for a paper accepted at the VDE Conference MaLeNe 2026

This repository contains the core machine learning and simulation pipelines for my thesis at the Hamburg University of Technology (TUHH). 

## 🚀 Project Overview
* **The Problem:** Conventional Geographic Greedy Routing (GGR) often fails in sparse and dynamic aeronautical networks (such as LDACS Air-to-Air communication) due to local minima and the memory effect.
* **The Solution:** Formulated the routing decision as a supervised Machine Learning task to predict or rank optimal next-hops using *only locally available information*.
* **The Data:** Generated datasets of local node features and optimal forwarding labels (based on Topological Advance) using both synthetic network snapshots and real-world OpenSky flight data.
* **The Models:** Trained and evaluated four ML variants: Logistic Regression, Random Forest, and LightGBM (in both classification and ranking variants).

## 🏆 Key Results
* **Custom Simulation:** Developed a dedicated simulation framework to evaluate the trained models under realistic Air-to-Air (A2A) network conditions.
* **Performance Leap:** Demonstrated that ML-based routing improves the success ratio by 24% compared to conventional GGR, particularly in highly sparse networks.
* **Scalability & Efficiency:** Achieved these performance gains while maintaining low computational complexity, proving that ML models can enable robust, scalable, and efficient routing strategies for future aeronautical communication systems.

![Success Ratio Comparison showing ML outperforming GGR](success_ratio.png)

---

## 📁 Repository Structure

### 1. Model Training (`Modeltraining/`)
Trains and evaluates the learning models using normalized feature sets.
* Contains standard classifiers (`LightGBM.py`, `LogReg.py`, `Random Forest.py`) and a pairwise-ranking variant (`LightGBM Ranker.py`).
* Implements model training pipelines.

### 2. Simulation & Evaluation (`Simulation/`)
Evaluates routing performance of the ML models against the greedy geographic routing (GGR) baseline.
* `simulation.py`: Generates multi-hop routes and logs hop stretch, success ratio, and time-per-path metrics.
* `plot_results.py`: Aggregates the simulation metrics and produces comparison charts.

---

## 🛠️ Code Showcase & Reproducibility

**Please Note:** This repository serves as a **code portfolio** to demonstrate the architectural approach and machine learning pipelines (model training & simulation) developed for my thesis. 

Due to ongoing academic research at the TUHH Institute of Communication Networks:
* Data generation scripts (e.g., OpenSky data fetching, synthetic topology generation) have been explicitly removed.
* Proprietary datasets and trained model weights (.joblib) are excluded.
* *The provided scripts are therefore for review purposes only and cannot be executed out-of-the-box without the internal datasets.*

*For inquiries regarding the methodology or publication, please reach out at kevin.nab@freenet.de.*
