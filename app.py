import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler

import time 

st.set_page_config(page_title="Loan Prediction Application", page_icon="🏦", layout="wide")

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'], errors='coerce')  
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at '{file_path}'. Please check the file path.")
        return None

file_path = 'Loan.csv'
df = load_data(file_path)

def show_home():
    """Home Page Content"""
    st.title("🏠 Home Page")
    st.markdown("""
    ### Welcome to the Loan Prediction Application!
    This application allows you to:
    - Understand the dataset
    - Perform Exploratory Data Analysis (EDA)
    - Visualize key insights
    - Build and evaluate machine learning models 
    """)

def show_about_dataset():
    """About Dataset Page Content"""
    st.title("📜 About the Dataset")
    if df is not None:
        st.markdown("### Loan Dataset Description")
        
        st.markdown("""This synthetic dataset comprises 20,000 records of personal and financial data, designed to facilitate the development of predictive models for risk assessment.
                    The dataset includes diverse features such as demographic information, credit history, employment status, income levels, existing debt, and other relevant financial metrics, providing a comprehensive foundation for sophisticated data-driven analysis and decision-making.""")
        with st.expander("Click to expand the dataset description", expanded=False):
            st.markdown("""
            This dataset contains information about loan applications, including:
            - **ApplicationDate**: Loan application date
            - **Age**: Applicant's age
            - **AnnualIncome**: Yearly income
            - **CreditScore**: Creditworthiness score
            - **EmploymentStatus**: Job situation
            - **EducationLevel**: Highest education attained
            - **Experience**: Work experience
            - **LoanAmount**: Requested loan size
            - **LoanDuration**: Loan repayment period
            - **MaritalStatus**: Applicant's marital state
            - **NumberOfDependents**: Number of dependents
            - **HomeOwnershipStatus**: Homeownership type
            - **MonthlyDebtPayments**: Monthly debt obligations
            - **CreditCardUtilizationRate**: Credit card usage percentage
            - **NumberOfOpenCreditLines**: Active credit lines
            - **NumberOfCreditInquiries**: Credit checks count
            - **DebtToIncomeRatio**: Debt to income proportion
            - **BankruptcyHistory**: Bankruptcy records
            - **LoanPurpose**: Reason for loan
            - **PreviousLoanDefaults**: Prior loan defaults
            - **PaymentHistory**: Past payment behavior
            - **LengthOfCreditHistory**: Credit history duration
            - **SavingsAccountBalance**: Savings account amount
            - **CheckingAccountBalance**: Checking account funds
            - **TotalAssets**: Total owned assets
            - **TotalLiabilities**: Total owed debts
            - **MonthlyIncome**: Income per month
            - **UtilityBillsPaymentHistory**: Utility payment record
            - **JobTenure**: Job duration
            - **NetWorth**: Total financial worth
            - **BaseInterestRate**: Starting interest rate
            - **InterestRate**: Applied interest rate
            - **MonthlyLoanPayment**: Monthly loan payment
            - **TotalDebtToIncomeRatio**: Total debt against income
            - **LoanApproved**: Loan approval status
            - **RiskScore**: Risk assessment score
            """)
        st.divider()
        st.markdown("### Dataset Overview")
        st.markdown("#### Select Application Year")
        df['ApplicationYear'] = df['ApplicationDate'].dt.year  
        unique_years = df['ApplicationYear'].unique()
        selected_years = st.multiselect("Select years:", sorted(unique_years))
        st.markdown("#### Select Education Level")
        unique_education_levels = df['EducationLevel'].unique()
        selected_education_levels = st.multiselect("Select education levels:", sorted(unique_education_levels))
        st.markdown("#### Select Age Range")
        age_min, age_max = st.slider("Select age range:", 18, 80, (18, 80))

        filtered_data = df.copy()
        if selected_years:
            filtered_data = filtered_data[filtered_data['ApplicationYear'].isin(selected_years)]
        if selected_education_levels:
            filtered_data = filtered_data[filtered_data['EducationLevel'].isin(selected_education_levels)]

        filtered_data = filtered_data[(filtered_data['Age'] >= age_min) & (filtered_data['Age'] <= age_max)]
        if not filtered_data.empty:
            st.write("Filtered Data:")
            st.write(filtered_data)
        else:
            st.write("No data available for the selected filters.")
    else:
        st.warning("Dataset not loaded. Please check the file path.")

def show_eda():
    """Exploratory Data Analysis Page Content"""
    st.title("📊 Exploratory Data Analysis (EDA)")
    if df is not None:

        st.markdown("### Dataset Shape")
        st.write("Number of rows:", df.shape[0])
        st.write("Number of columns:", df.shape[1])

        st.markdown("### Data Types")
        data_types_df = df.dtypes.reset_index()  
        data_types_df.columns = ['Column Name', 'Data Type']  
        st.dataframe(data_types_df)  

        st.markdown("### Missing Values")
        st.write(df.isnull().sum())
        missing_count = df.isnull().sum().sum()
        st.write(f"Number of missing values: {missing_count}")

        st.markdown("### Duplicate Records")
        duplicate_count = df.duplicated().sum()
        st.write(f"Number of duplicate records: {duplicate_count}")

        st.markdown("### Dataset Summary")
        st.write(df.describe())

        st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  
        st.markdown("<h2 style='text-align: center;'>🔢 Analysis of Numerical Variables</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  

        st.markdown("### Loan Approved Distribution")
        st.write("ℹ️LoanApproved is the target variable that indicates whether a loan application was approved or rejected.")
        
        loan_approval_counts = df['LoanApproved'].value_counts()
        total_count = loan_approval_counts.sum()
        loan_approval_percentages = (loan_approval_counts / total_count) * 100

        distribution_df = pd.DataFrame({
            'Count': loan_approval_counts,
            'Percentage': loan_approval_percentages
        }).reset_index()

        distribution_df.columns = ['Loan Approved', 'Count', 'Percentage']

        st.write(distribution_df)

        st.bar_chart(distribution_df.set_index('Loan Approved')['Percentage'])

        st.divider()
        st.markdown(f"""
                    ⬇️ Interpretation ⬇️
                    
                    Approximately **76.1%** of loan applications were **rejected** and **23.9%** were **approved**.
                    These results show that the dataset is **imbalanced**, with a higher number of rejected applications.
                     """)
        st.divider()

        st.markdown("### Select a Numerical Variable to Visualize")
        numerical_cols = [
            'Age', 'AnnualIncome', 'CreditScore', 'Experience', 'LoanAmount',
            'MonthlyDebtPayments', 'CreditCardUtilizationRate', 'DebtToIncomeRatio',
            'SavingsAccountBalance', 'CheckingAccountBalance', 'TotalAssets',
            'TotalLiabilities', 'MonthlyIncome', 'NetWorth', 'BaseInterestRate',
            'InterestRate', 'MonthlyLoanPayment', 'JobTenure', 'RiskScore', 
            'LengthOfCreditHistory', 'LoanDuration', 'PaymentHistory'
        ]

        selected_col = st.selectbox("Select a variable:", numerical_cols)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df[selected_col], kde=True, bins=30, ax=ax[0])
        ax[0].set_title(f'Distribution of {selected_col}')
        sns.boxplot(y=df[selected_col], ax=ax[1])
        ax[1].set_title(f'Boxplot of {selected_col}')
        st.pyplot(fig)

        skewness_value = df[selected_col].skew()
        kurtosis_value = df[selected_col].kurtosis()

        st.markdown("#### ⬇️ Skewness and Kurtosis ⬇️")
        
        def interpret_skewness(skewness):
            if skewness < -1:
                return f"Highly negatively skewed (Skewness: {skewness:.2f})" 
            elif -1 <= skewness < 0:
                return f"Moderately negatively skewed (Skewness: {skewness:.2f})"
            elif 0 < skewness < 1:
                return f"Moderately positively skewed (Skewness: {skewness:.2f})"
            elif skewness >= 1:
                return f"Highly positively skewed (Skewness: {skewness:.2f})"
            else:
                return "Symmetrical"

        interpretation = interpret_skewness(skewness_value)
        st.write(f"Skewness ({selected_col}): {interpretation}")

        def interpret_kurtosis(kurtosis):
            if kurtosis > 3:
                return f"The distribution is leptokurtic (sharp peak), with heavy tails and outliers.  It is recommended to investigate the data further (Kurtosis: {kurtosis:.2f})"
            elif kurtosis > 0:
                return f"The distribution is platykurtic (flatter peak), with lighter tails and less outliers. A more uniform distribution (Kurtosis: {kurtosis:.2f}) "
            elif kurtosis == 0:
                return f"The distribution is mesokurtic, which means it follows a normal distribution shape (no heavy tails or outliers)  (Kurtosis: {kurtosis:.2f})"
            else:
                return f"The distribution is platykurtic with extremely light tails and very few outliers (Kurtosis: {kurtosis:.2f})"

        interpretation_kurtosis = interpret_kurtosis(kurtosis_value)
        st.write(f"Kurtosis ({selected_col}): {interpretation_kurtosis}")

        st.markdown("#### ⬇️ Outliers Detection ⬇️")

        outliers = df[(df[selected_col] < df[selected_col].quantile(0.25) - 1.5 * (df[selected_col].quantile(0.75) - df[selected_col].quantile(0.25))) |
                       (df[selected_col] > df[selected_col].quantile(0.75) + 1.5 * (df[selected_col].quantile(0.75) - df[selected_col].quantile(0.25)))]
        outlier_count = len(outliers)

        st.write(f"⚠️{outlier_count} potential outliers")
        
        st.divider()

        with st.expander("⬇️ Key Observations ⬇️", expanded=False):
            st.markdown("""

                        🗨️**VARIABLES WITHOUT OUTLIERS**
                        - **LengthOfCreditHistory** and **LoanDuration** have no outliers, suggesting these distributions are well-behaved or within expected ranges.
                    
                        🗨️**VARIABLES WITH MANY OUTLIERS**
                        - Variables like **SavingsAccountBalance**, **CheckingAccountBalance**, and **NetWorth** have a high number of potential outliers.

                        🗨️**MODERATE OUTLIERS**
                        - **Age**, **RiskScore**, and **CreditCardUtilizationRate** have fewer outliers, which might be natural variability.
                    
                        """)
        
        st.divider()

        st.markdown("<h2 style='text-align: center;'>Informations about Log Transformation</h2>", unsafe_allow_html=True)
        st.markdown("""
                     
                    In our analysis, we have identified certain variables that exhibit a skewed distribution. To improve the interpretability of the data and reduce the impact of extreme values, we will apply a log transformation to the following variables:
                    - **AnnualIncome**
                    - **LoanAmount**
                    - **MonthlyDebtPayments**
                    - **CreditCardUtilizationRate**
                    - **DebtToIncomeRatio**
                    - **SavingsAccountBalance**
                    - **CheckingAccountBalance**
                    - **TotalAssets**
                    - **TotalLiabilities**
                    - **MonthlyIncome**
                    - **NetWorth**
                    - **MonthlyLoanPayment**
                    
                    This transformation will help us achieve a more normal distribution of the data, thereby facilitating further analysis and modeling.
                """)

        columns_to_transform = [
            'AnnualIncome', 'LoanAmount', 'MonthlyDebtPayments', 
            'CreditCardUtilizationRate', 'DebtToIncomeRatio', 
            'SavingsAccountBalance', 'CheckingAccountBalance', 
            'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 
            'NetWorth', 'MonthlyLoanPayment'
        ]

        df_transformed = df.copy()
        for col in columns_to_transform:
            df_transformed[col] = np.log1p(df_transformed[col])  

        st.markdown(f"#### Select a Numerical Variable to Visualize (After Log Transformation)")
        numerical_cols = [
            'AnnualIncome', 'LoanAmount', 'MonthlyDebtPayments', 
            'CreditCardUtilizationRate', 'DebtToIncomeRatio', 
            'SavingsAccountBalance', 'CheckingAccountBalance', 
            'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 
            'NetWorth', 'MonthlyLoanPayment'
        ]

        selected_col = st.selectbox("Select a variable:", numerical_cols, key="select_variable")

        if selected_col:
            n_cols = 2
            n_rows = 1  

            plt.figure(figsize=(15, 5))

            plt.subplot(n_rows, n_cols, 1)
            sns.histplot(df_transformed[selected_col] if selected_col in columns_to_transform else df[selected_col], kde=True)
            plt.title(f'{selected_col} Distribution (After Log Transformation)' if selected_col in columns_to_transform else f'{selected_col} Distribution')

            plt.subplot(n_rows, n_cols, 2)
            sns.boxplot(y=df_transformed[selected_col] if selected_col in columns_to_transform else df[selected_col])
            plt.title(f'{selected_col} (After Log Transformation)' if selected_col in columns_to_transform else selected_col)

            plt.tight_layout()

            st.pyplot(plt)

        st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  
        st.markdown("<h2 style='text-align: center;'>🔢 Analysis of Categorical Variables</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  

        categorical = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']

        selected_col = st.selectbox('Select a categorical variable to visualize:', categorical)

        st.write(f'Value counts for {selected_col}:')
        st.write(df[selected_col].value_counts())

        plt.figure(figsize=(10, 5))
        sns.countplot(y=selected_col, data=df)
        plt.title(f'Distribution of {selected_col}')

        st.pyplot(plt)

        plt.clf()


        st.markdown("<h2 style='text-align: center;'>Data Pre-Processing</h2>", unsafe_allow_html=True)
        st.markdown(f"""
                    In this step we prepare the data for further analysis and training of the machine learning model. 
                    - To transform **categorical** variables into **numerical** values, we will apply the **Label Encoding** technique.
                    - To **standardize** the data to have a mean of 0 and a standard deviation of 1 we will apply **Standard Scaling**.
                    """)
        label_encoder = LabelEncoder()
        for col in categorical:
            df_transformed[col] = label_encoder.fit_transform(df_transformed[col])
        
        scaler = StandardScaler()
        df_transformed[numerical_cols] = scaler.fit_transform(df_transformed[numerical_cols])
        st.write("Transformed Data:")
        st.dataframe(df_transformed)

        numerical_cols3 = [
            'Age', 'AnnualIncome', 'CreditScore', 'Experience', 'LoanAmount',
            'MonthlyDebtPayments', 'CreditCardUtilizationRate', 'DebtToIncomeRatio',
            'SavingsAccountBalance', 'CheckingAccountBalance', 'TotalAssets',
            'TotalLiabilities', 'MonthlyIncome', 'NetWorth', 'BaseInterestRate',
            'InterestRate', 'MonthlyLoanPayment', 'JobTenure', 'RiskScore', 
            'LengthOfCreditHistory', 'LoanDuration', 'PaymentHistory'
        ]
        st.markdown("#### Correlation Matrix")
        corr_matrix = df[numerical_cols3].corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)

        highly_correlated_variables = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    highly_correlated_variables.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        if highly_correlated_variables:
            st.markdown("##### Highly Correlated Variables(correlation > 0.7):")
            # st.write("Highly correlated variables (correlation > 0.7):")
            for var1, var2, corr_value in highly_correlated_variables:
                st.write(f"**{var1}** and **{var2}** | Correlation: {corr_value:.2f}")
        else:
            st.write("No highly correlated variables found (correlation > 0.7).")
        
        st.divider()

        correlation_with_target = df_transformed.corr()['LoanApproved'].sort_values(ascending=False)
           
        st.write("#### Correlation with 'LoanApproved':")
        for variable, correlation in correlation_with_target.items():
            st.write(f"**Variable:** {variable} | **Correlation:** {correlation:.2f}")

        st.markdown("<h2 style='text-align: center;'>Future Updates Coming Soon!</h2>", unsafe_allow_html=True)
        progress_bar = st.progress(0)  
        # # Simulate progress
        # for i in range(1, 101):
        #     time.sleep(0.05)  # Simulate some work being done
        #     progress_bar.progress(i)  # Update the progress bar

        # st.success("Project will have more features soon!")


def display_machine_learning_techniques():
    st.title("🤖 Machine Learning Techniques")
    st.markdown("### Introduction")
    st.write("On this page, we will explore and demonstrate several powerful Machine Learning models applied to loan approval data. These models include Logistic Regression, Random Forest, Naive Bayes, K-Nearest Neighbors, and Support Vector Machines (SVM). Each model will be evaluated based on its performance metrics to predict the likelihood of a loan approval, helping to understand the strengths and nuances of each technique in tackling this problem.")
    
    # Dropdown
    model_option = st.selectbox("Select a Model:", 
                                ["Logistic Regression", "Random Forest", "Naive Bayes", "K-Nearest Neighbors", "Decision Tree", "Neural Network (NLP)"])

    # Features and target
    X = df[['MonthlyIncome', 'NetWorth', 'CreditScore', 'Age', 'PreviousLoanDefaults', 'TotalDebtToIncomeRatio', 'MonthlyDebtPayments', 'RiskScore']]
    y = df['LoanApproved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    if model_option == "Logistic Regression":
        st.markdown("<h2 style='text-align: center;'> Logistic Regression</h2>", unsafe_allow_html=True)
        
        log_reg_model = LogisticRegression()
        log_reg_model.fit(X_train, y_train)
        y_pred = log_reg_model.predict(X_test)

        st.markdown("#### Model Evaluation:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write("**Accuracy:**", accuracy)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("**Classification Report:**")
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))
        st.markdown("#### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(25, 10))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.markdown("#### ROC Curve & AUC:")
        y_prob = log_reg_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(25, 10))
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        auc_score = roc_auc_score(y_test, y_prob)
        st.write("**AUC Score:**", auc_score)

        with st.expander("Click here to see interpretation"):
            st.markdown("### Model Interpretation for Logistic Regression")
            st.markdown("#### **Accuracy:**")
            st.write("The model has an accuracy of **94.28%**, meaning 94.28% of its predictions were correct on the test set.")
            st.markdown("#### **Classification Report**:")
            st.write("""
            **Precision** represents the accuracy of the positive predictions for each class (how many of the positive predictions were correct).
            **Recall** measures the model's ability to correctly identify positive cases (how many of the actual positives were detected correctly).
            **F1-Score** is the harmonic mean between precision and recall, providing a balanced measure of the model's performance.
            **Support** represents the number of instances for each class.

            - **Class 0 (Loan Rejected):**
                - **Precision:** 0.95 — 95% of the loan rejection predictions were correct.
                - **Recall:** 0.97 — 97% of the actual loan rejection cases were correctly identified.
                - **F1-Score:** 0.96 — Overall performance for this class is very good.
                
            - **Class 1 (Loan Approved):**
                - **Precision:** 0.91 — 91% of the loan approval predictions were correct.
                - **Recall:** 0.85 — 85% of the actual loan approval cases were correctly identified.
                - **F1-Score:** 0.88 — Performance for this class is good, but there's room for improvement.

            - **Macro Avg (Arithmetic average of classes):**
                - **Precision:** 0.93 — The model has a good average precision across all classes.
                - **Recall:** 0.91 — The average recall across all classes is good.
                - **F1-Score:** 0.92 — Overall performance is excellent.

            - **Weighted Avg (Weighted average):**
                - **Precision:** 0.94 — The model has good overall precision, taking into account the number of examples in each class.
                - **Recall:** 0.94 — The global detection rate is similar.
                - **F1-Score:** 0.94 — Overall performance is consistent.
            """)
            st.markdown("#### **Feature Importance**:")
            st.write("""
            - **RiskScore:**  -0.38 — The most important predictor for our model. A higher **RiskScore** decreases the likelihood of loan approval.
            - **CreditScore:** 0.02 — A higher credit score increases the chances of loan approval.
            - **Age:** 0.004 — Age has a minor influence.
            - **MonthlyIncome:** 0.0007 — Monthly income has a very low importance.
            - **NetWorth:** 0.000003 — Net worth has negligible impact.
            - **MonthlyDebtPayments:** -0.0009 — Monthly debt payments have a small negative impact.
            - **PreviousLoanDefaults:** -0.002 — Previous defaults negatively affect approval likelihood.
            - **TotalDebtToIncomeRatio:** -0.004 — Debt-to-income ratio has a small negative impact.
            """)
            st.markdown("#### **AUC Score:**")
            st.write(f"**AUC Score:** {0.985:.2f} — An AUC of **0.985** indicates excellent performance in distinguishing between the two classes (rejected/approved).")

    # Random Forest
    elif model_option == "Random Forest":
        st.markdown("<h2 style='text-align: center;'> Random Forest</h2>", unsafe_allow_html=True)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        st.markdown("#### Model Evaluation:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write("**Accuracy:**", accuracy)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("**Classification Report:**")
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

        st.markdown("#### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(25, 10))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.markdown("#### ROC Curve & AUC:")
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(25, 10))
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        auc_score = roc_auc_score(y_test, y_prob)
        st.write("**AUC Score:**", auc_score)

    # Naive Bayes
    elif model_option == "Naive Bayes":
        st.markdown("<h2 style='text-align: center;'> Naive Bayes</h2>", unsafe_allow_html=True)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)

        st.markdown("#### Model Evaluation:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write("**Accuracy:**", accuracy)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("**Classification Report:**")
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

        st.markdown("#### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(25, 10))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.markdown("#### ROC Curve & AUC:")
        y_prob = gnb.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(25, 10))
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        auc_score = roc_auc_score(y_test, y_prob)
        st.write("**AUC Score:**", auc_score)

    # K-Nearest Neighbors
    elif model_option == "K-Nearest Neighbors":
        st.markdown("<h2 style='text-align: center;'> K-Nearest Neighbors</h2>", unsafe_allow_html=True)
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        st.markdown("#### Model Evaluation:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write("**Accuracy:**", accuracy)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("**Classification Report:**")
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))
        st.markdown("#### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(25, 10))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.markdown("#### ROC Curve & AUC:")
        y_prob = knn_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(25, 10))
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        auc_score = roc_auc_score(y_test, y_prob)
        st.write("**AUC Score:**", auc_score)

    # Decision Tree
    elif model_option == "Decision Tree":
        st.markdown("<h2 style='text-align: center;'> Decision Tree</h2>", unsafe_allow_html=True)
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)

        st.markdown("#### Model Evaluation:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write("**Accuracy:**", accuracy)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("**Classification Report:**")
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

        st.markdown("#### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(25, 10))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        st.markdown("#### ROC Curve & AUC:")
        y_prob = dt_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(25, 10))
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        auc_score = roc_auc_score(y_test, y_prob)
        st.write("**AUC Score:**", auc_score)

    # Neural Network (MLP)
    elif model_option == "Neural Network (NLP)":
        st.markdown("<h2 style='text-align: center;'> Neural Network (NLP)</h2>", unsafe_allow_html=True)
        mlp_model = MLPClassifier(random_state=42)
        mlp_model.fit(X_train, y_train)
        y_pred = mlp_model.predict(X_test)

        st.markdown("#### Model Evaluation:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write("**Accuracy:**", accuracy)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("**Classification Report:**")
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))
        st.markdown("#### Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(25, 10))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        st.markdown("#### ROC Curve & AUC:")
        y_prob = mlp_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(25, 10))
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        auc_score = roc_auc_score(y_test, y_prob)
        st.write("**AUC Score:**", auc_score)

def main():
    """Main Application Function"""
    st.sidebar.title("📌Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About Dataset", "EDA", "Machine Learning"])

    if page == "Home":
        show_home()
    elif page == "About Dataset":
        show_about_dataset()
    elif page == "EDA":
        show_eda()
    elif page == 'Machine Learning':
        display_machine_learning_techniques()

if __name__ == "__main__":
    main()