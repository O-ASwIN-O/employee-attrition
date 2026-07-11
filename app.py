import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

model          = joblib.load('xgb_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.set_page_config(page_title="Employee Attrition Risk", layout="wide")
st.title("Employee Resignation Early Warning System")

# ── Shared preprocessing function ─────────────────────────────────────────────
def preprocess(df):
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if 'Attrition' in df.columns:
        df = df.drop(columns=['Attrition'])

    df['Gender']        = df['Gender'].map({'Male': 1, 'Female': 0})
    df['OverTime']      = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['BusinessTravel'] = df['BusinessTravel'].map({
        'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2
    })
    df = pd.get_dummies(df, columns=['Department', 'JobRole', 'MaritalStatus'], dtype=int)
    df['SatisfactionScore'] = (
        df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']
    ) / 3
    df['IncomePerYear'] = df['MonthlyIncome'] * 12 / (df['YearsAtCompany'] + 1)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# ── SHAP explainer (cached so it doesn't reload every time) ───────────────────
@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

# ── Two tabs ──────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single employee prediction", "Bulk CSV upload"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single employee form
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter employee details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age             = st.number_input("Age",                      min_value=18, max_value=65, value=30)
        monthly_income  = st.number_input("Monthly income",           min_value=1000, max_value=20000, value=5000)
        years_company   = st.number_input("Years at company",         min_value=0, max_value=40, value=3)
        years_role      = st.number_input("Years in current role",    min_value=0, max_value=20, value=2)

    with col2:
        job_satisfaction   = st.selectbox("Job satisfaction",          [1, 2, 3, 4], index=1,
                                          format_func=lambda x: {1:"1 - Low",2:"2 - Medium",3:"3 - High",4:"4 - Very High"}[x])
        env_satisfaction   = st.selectbox("Environment satisfaction",  [1, 2, 3, 4], index=2,
                                          format_func=lambda x: {1:"1 - Low",2:"2 - Medium",3:"3 - High",4:"4 - Very High"}[x])
        rel_satisfaction   = st.selectbox("Relationship satisfaction", [1, 2, 3, 4], index=2,
                                          format_func=lambda x: {1:"1 - Low",2:"2 - Medium",3:"3 - High",4:"4 - Very High"}[x])
        work_life_balance  = st.selectbox("Work-life balance",         [1, 2, 3, 4], index=2,
                                          format_func=lambda x: {1:"1 - Bad",2:"2 - Good",3:"3 - Better",4:"4 - Best"}[x])

    with col3:
        overtime        = st.selectbox("Overtime",          ["Yes", "No"])
        gender          = st.selectbox("Gender",            ["Male", "Female"])
        department      = st.selectbox("Department",        ["Sales", "Research & Development", "Human Resources"])
        job_role        = st.selectbox("Job role",          ["Sales Executive","Research Scientist","Laboratory Technician",
                                                             "Manufacturing Director","Healthcare Representative",
                                                             "Manager","Sales Representative","Research Director","Human Resources"])
        marital_status  = st.selectbox("Marital status",   ["Single", "Married", "Divorced"])
        business_travel = st.selectbox("Business travel",  ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

    # remaining numeric fields with defaults
    distance        = st.slider("Distance from home (km)", 1, 29, 10)
    num_companies   = st.slider("Number of companies worked", 0, 9, 2)
    stock_option    = st.selectbox("Stock option level", [0, 1, 2, 3])
    performance     = st.selectbox("Performance rating", [3, 4], format_func=lambda x: {3:"3 - Excellent", 4:"4 - Outstanding"}[x])
    education       = st.selectbox("Education", [1,2,3,4,5],
                                   format_func=lambda x:{1:"Below college",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])

    if st.button("Predict attrition risk", type="primary"):

        # Build a single-row dataframe matching training structure
        input_dict = {
            "Age": age, "DailyRate": 800, "DistanceFromHome": distance,
            "Education": education, "EducationField": "Life Sciences",
            "EmployeeNumber": 9999,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender, "HourlyRate": 66,
            "JobInvolvement": 3, "JobLevel": 2,
            "JobRole": job_role, "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
            "MonthlyRate": 14000, "NumCompaniesWorked": num_companies,
            "OverTime": overtime, "PercentSalaryHike": 14,
            "PerformanceRating": performance,
            "RelationshipSatisfaction": rel_satisfaction,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": years_company + 3,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
            "Department": department, "BusinessTravel": business_travel,
        }

        input_df  = pd.DataFrame([input_dict])
        processed = preprocess(input_df)
        prob      = model.predict_proba(processed)[0][1]
        risk_pct  = round(prob * 100, 1)

        # Risk level label
        if prob >= 0.6:
            level, color = "High risk", "red"
        elif prob >= 0.3:
            level, color = "Medium risk", "orange"
        else:
            level, color = "Low risk", "green"

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Attrition probability", f"{risk_pct}%")
        m2.metric("Risk level", level)

        # SHAP waterfall for this employee
        st.subheader("Why this prediction? (SHAP)")
        shap_exp = explainer(processed)
        fig, ax  = plt.subplots()
        shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk CSV upload
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload employee CSV")
    st.caption("File must have the same columns as the IBM HR dataset.")

    uploaded = st.file_uploader("Choose CSV file", type="csv")

    if uploaded:
        raw_df    = pd.read_csv(uploaded)
        processed = preprocess(raw_df.copy())
        probs     = model.predict_proba(processed)[:, 1]

        raw_df['Risk Score (%)'] = (probs * 100).round(1)
        raw_df['Risk Level']     = pd.cut(
            probs,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("High risk",   int((raw_df['Risk Level'] == 'High').sum()))
        c2.metric("Medium risk", int((raw_df['Risk Level'] == 'Medium').sum()))
        c3.metric("Low risk",    int((raw_df['Risk Level'] == 'Low').sum()))

        st.divider()

        # Filterable table
        filter_level = st.selectbox("Filter by risk level", ["All", "High", "Medium", "Low"])
        display_df   = raw_df if filter_level == "All" else raw_df[raw_df['Risk Level'] == filter_level]

        show_cols = [c for c in ['EmployeeNumber','Age','JobRole','Department',
                                  'MonthlyIncome','OverTime','Risk Score (%)','Risk Level']
                     if c in display_df.columns]
        st.dataframe(
            display_df[show_cols].sort_values('Risk Score (%)', ascending=False),
            use_container_width=True
        )

        # SHAP for the single most at-risk employee
        st.divider()
        st.subheader("Top at-risk employee — SHAP explanation")
        top_idx  = int(probs.argmax())
        shap_exp = explainer(processed)
        fig, ax  = plt.subplots()
        shap.plots.waterfall(shap_exp[top_idx], max_display=10, show=False)
        st.pyplot(fig)
        plt.close()

        # Global feature importance
        st.subheader("Global feature importance")
        fig2, ax2 = plt.subplots()
        shap.summary_plot(explainer.shap_values(processed), processed, max_display=15, show=False)
        st.pyplot(fig2)
        plt.close()

        # Download button
        st.divider()
        csv_out = display_df[show_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv_out,
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )