# Employee Resignation Early Warning System

> Predict which employees are likely to leave in the next 3–6 months — and explain exactly why.

🔴 **Live Demo** → [employee-attrition.streamlit.app](https://employee-attrition-bl32xehex7ngteru5d3q7u.streamlit.app/)

---

## What this project does

HR teams lose thousands of dollars every time a trained employee leaves. This system uses machine learning to flag at-risk employees **before** they resign — and uses SHAP explainability to tell HR exactly which factors are driving the risk for each individual person.

Instead of just saying *"this employee might leave"*, it says:

> *"Employee #1042 is 82% likely to leave — primarily because they work overtime, earn below the department average, and have low job satisfaction."*

---

## Demo

| Single employee prediction | Bulk CSV prediction |
|---|---|
| Fill a form with one employee's details | Upload your entire HR CSV |
| Get instant risk score + SHAP explanation | Get risk scores for every employee |
| See exactly which factors drove the prediction | Filter by High / Medium / Low risk |

---

## Tech stack

| Layer | Tools |
|---|---|
| Machine learning | XGBoost, scikit-learn, imbalanced-learn (SMOTE) |
| Explainability | SHAP (waterfall + summary plots) |
| Experiment tracking | MLflow |
| Frontend | Streamlit |
| Containerization | Docker |
| Deployment | Streamlit Cloud |
| Language | Python 3.11 |

---

## ML pipeline

```
Raw HR data (1,470 employees, 35 features)
        │
        ▼
Phase 1 ── EDA & class imbalance discovery (16% attrition rate)
        │
        ▼
Phase 2 ── Feature engineering
           · Encode: Gender, OverTime, BusinessTravel
           · One-hot: Department, JobRole, MaritalStatus
           · Derived: SatisfactionScore, IncomePerYear
        │
        ▼
Phase 3 ── Model training
           · Baseline: Logistic Regression
           · Ensemble: Random Forest
           · Final:    XGBoost + SMOTE (fixes class imbalance)
        │
        ▼
Phase 4 ── Evaluation
           · AUC-ROC, Precision, Recall, F1
           · Confusion matrix
           · Focus on recall for attrition class
        │
        ▼
Phase 5 ── MLflow experiment tracking
           · Log params, metrics, artifacts per run
           · Register best model to Model Registry
        │
        ▼
Phase 6 ── SHAP explainability
           · Global: which features matter most overall
           · Local:  why this specific employee is flagged
        │
        ▼
Phase 7 ── Streamlit app (single + bulk prediction)
        │
        ▼
Phase 8 ── Docker + Streamlit Cloud deployment
```

---

## Key results

| Metric | Score |
|---|---|
| AUC-ROC | ~0.84 |
| Recall (attrition class) | ~0.72 |
| Precision (attrition class) | ~0.51 |

> **Why recall matters more than accuracy here:** A model that predicts "No" for every employee gets 84% accuracy but catches zero people who actually leave. We optimize for recall — catching real leavers — even at the cost of some false positives.

---

## Top attrition drivers (SHAP)

Based on the trained model, the strongest predictors of an employee leaving are:

1. **OverTime** — working overtime is the single strongest signal
2. **MonthlyIncome** — lower income = significantly higher risk
3. **Age** — younger employees (25–35) leave most frequently
4. **YearsAtCompany** — first 1–3 years are highest risk
5. **JobSatisfaction** — low satisfaction strongly predicts attrition
6. **StockOptionLevel** — no stock options = higher risk
7. **MaritalStatus_Single** — single employees are more mobile

---

## Run locally

### Option 1 — Standard Python

```bash
# Clone the repo
git clone https://github.com/O-ASwIN-O/employee-attrition.git
cd employee-attrition

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Option 2 — Docker

```bash
# Make sure Docker Desktop is running, then:
docker build -t attrition-app .
docker run -p 8501:8501 attrition-app
```

Open `http://localhost:8501` in your browser.

---

## Project structure

```
employee-attrition/
├── app.py                        # Streamlit app (single + bulk prediction)
├── Employee_resignation.ipynb    # Full ML pipeline (Phases 1–6)
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**
- 1,470 employees · 35 features · 16% attrition rate
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Synthetic dataset created by IBM data scientists for HR analytics research

---

## What I learned

- How class imbalance silently breaks ML models — and how SMOTE fixes it
- Why accuracy is a misleading metric for imbalanced classification
- How SHAP values translate model predictions into human-readable business reasons
- MLflow experiment tracking for reproducible ML
- Containerizing a data science app with Docker
- Full deployment pipeline from local notebook to live production URL

---

## Author

**Aswin**
[GitHub](https://github.com/O-ASwIN-O)