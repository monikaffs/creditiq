# 💳 CreditIQ — Loan Default Predictor

> An ML web app that predicts loan default probability and explains 
> decisions using SHAP — built for transparent credit risk assessment.

🔗 **[Live Demo](your-streamlit-url)**

---

## 📋 Main Bullet Point
* **Developed a Credit Risk Prediction web app** using XGBoost + SHAP explainability on 150K records (*Give Me Some Credit*, Kaggle); achieved **AUC-ROC of 86.95%**; deployed on Streamlit Cloud with batch prediction and threshold tuning features.

## 🔑 Key Features
| Topic | Key point |
|---|---|
| **Why XGBoost?** | Handles missing values, imbalanced data, and tabular interactions efficiently |
| **Class imbalance** | Used `scale_pos_weight` (neg/pos ratio) instead of oversampling |
| **SHAP** | Game-theoretic feature attributions for each prediction |
| **Why not accuracy?** | 93% of samples are non-default — accuracy is misleading |
| **Business impact** | Flag high-risk applicants before approval to reduce losses |

---

## 📸 Screenshots
*[Screenshots to be added: Home, Risk Predictor with gauge, Model Performance]*

---

## 🎯 What it does
- Predicts default probability for individual applicants
- Batch prediction via CSV upload
- SHAP explainability for every prediction
- Model comparison: Logistic Regression vs Random Forest vs XGBoost
- Interactive threshold tuner

## 📊 Model Performance
| Model | Accuracy | Recall | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 76.12% | 74.26% | 83.47% |
| Random Forest | 84.01% | 71.12% | 86.76% |
| XGBoost ✅ | 81.05% | 76.31% | 86.95% |

## 🛠 Tech Stack
Python · XGBoost · SHAP · Streamlit · Pandas · Scikit-learn · Plotly

## 📁 Dataset
Give Me Some Credit — Kaggle (150,000 records, 11 features)

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

## 👩‍💻 Built by
Monika Dangi · BCA, SICSR Pune
