# 🧠 PD Speech Classification

This repository contains the code and experiments for a research project on **Parkinson’s Disease (PD) classification from speech features**.  
The goal is to explore different *Machine Learning* algorithms and dimensionality reduction techniques to support the **early diagnosis** of PD.

---

## ⚙️ Technologies
- Python 3.10+
- Pandas, NumPy, Scikit-learn  
- Matplotlib, Seaborn  
- Imbalanced-learn (SMOTE)  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
Install dependencies:

bash
Copiar código
pip install -r requirements.txt
Run the main script:

bash
Copiar código
python run_loso_pd.py
📊 What the Code Does
Preprocesses speech data

Leave-One-Subject-Out (LOSO) cross-validation

Balancing with SMOTE

Trains base models (Logistic Regression, SVM, Random Forest)

Visualizations: PCA, t-SNE, LDA, ROC and Precision-Recall curves, Confusion Matrix

Feature importance analysis

📂 Repository Structure
bash
Copiar código
├── data/          # Dataset
├── outputs/       # Results (plots, metrics)
├── run_loso_pd.py # Main script
└── README.md
✍️ Developed by Luana Dantas Pontes Espínola Casado
🎓 Master’s Student in Computer Science - UFERSA

