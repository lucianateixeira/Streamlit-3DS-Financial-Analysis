import pandas as pd
import streamlit as st
from math import pi
import numpy as np
import joblib  # For loading the .pkl models
import random
import datetime  # For date handling
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from PIL import Image  # For loading images

import warnings
warnings.filterwarnings("ignore")

# Load the actual datasets
scadata = pd.read_csv('scadata.csv')
cleaned_data = pd.read_csv('cleaned_data.csv')

# Define your custom data dictionary with feature descriptions
data_dictionary = {
    'scadata': {
        'AUTH_CDT': 'Date of the transaction.',
        'MERCH_CAT_CD': 'Merchant Category Code - each code corresponds to a specific market segment.',
        'TRAN_FRAUD': 'Fraud indicator - Y (yes), N (no), and U (undefined).',
        'ZIP_CD': 'Zip Code.',
        'ECI_IND': 'Electronic Commerce Indicator (ECI).',
        'APPROVED_AMT': 'Approved amount of the transaction.',
        'DECLINE_RES1': 'Decline reason code (company specific).',
        'AUTH_CD': 'Authorization code (unique value - company specific).',
        'TSYS_PROD_CD': 'Internal code (unique value - company specific).',
        'ADFLAG': 'Final result (approved or declined).'
    },
    'cleaned_data': {
        'AUTH_CDT': 'Date of the transaction.',
        'MERCH_CAT_CD': 'Merchant Category Code - each code corresponds to a specific market segment.',
        'TRAN_FRAUD': 'Fraud indicator - Y (yes), N (no), and U (undefined).',
        'ZIP_CD': 'Zip Code.',
        'ECI_IND': 'Electronic Commerce Indicator (ECI).',
        'APPROVED_AMT': 'Approved amount of the transaction.',
        'DECLINE_RES1': 'Decline reason code (company specific).',
        'AUTH_CD': 'Authorization code (unique value - company specific).',
        'ADFLAG': 'Final result (approved or declined).',
        'MCC_Description': 'Description of the Merchant Category Code.',
        'ECI_Description': 'Description of Electronic Commerce Indicator.',
        'TRAN_FRAUD_N': 'Non-Fraudulent Transaction.',
        'TRAN_FRAUD_Y': 'Fraudulent Transaction.',
        'ADFLAG_BINARY': 'Final result (approved or declined - 0/1).'
    }
}

# Apply Google Font and custom CSS
google_font = '<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;700&display=swap" rel="stylesheet">'
custom_css = """
<style>
    /* Target the active menu item in streamlit-option-menu */
    .nav-link.active {
        background-color: #BFCC94 !important;  /* Set background color for the active item */
        color: white !important;               /* Set text color for the active item */
    }
    .nav-link:hover {
        background-color: #BFCC94 !important;  /* Ensure hover state matches active state */
        color: white !important;
    }
</style>
"""
st.markdown(google_font, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    # Display the logo at the top of the sidebar
    logo = Image.open('logo.png')
    st.image(logo, use_column_width=True)

    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Dataset Information", "Training Results", "Models"],
        icons=["house", "info-circle", "clipboard", "gear"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("""
    <div class="info-box">
        <p class="name">Luciana Teixeira</p>
        <p class="institution">MSc Data Analytics | 2021322</p>
        <p><a href="https://github.com/LucianaTeixeira322" class="link">GitHub</a> | <a href="https://www.cct.ie/" class="link"> CCT College</a></p>
    </div>
    """, unsafe_allow_html=True)

# Main body content
if selected == "Home":
    st.header('Harnessing the power of data: Enhancing e-commerce transaction security through analytics')
    st.write("""
        In the rapidly evolving landscape of digital commerce, the increase in online shopping and payments has led to a significant rise in Card-Not-Present (CNP) transactions, which are particularly vulnerable to online fraud. 
        To mitigate these risks, security protocols such as Strong Customer Authentication (SCA) and 3D Secure (3DS) were introduced. While effective in enhancing security, these systems often result in friction during the checkout process, negatively impacting the user experience.
        This research addresses the pressing need to optimize fraud detection in CNP transactions by exploring the potential of machine learning to improve the 3DS process. A mixed-methodology approach was employed, combining qualitative insights from financial experts with quantitative analysis of transaction data. 
        Machine learning models, including Random Forest, Logistic Regression, Support Vector Classifiers, and Multilayer Perceptrons, were developed to predict transaction outcomes, aiming to reduce unnecessary authentication challenges without compromising security.

        ### Objectives
        - To identify and prioritise the key determinants shaping banks' risk appetites and security expectations for online purchases.
        - To conduct a data analysis on critical transaction information and patterns in order to determine the probability of successful authentication irrespective of authentication challenges.
        - To apply predictive modeling techniques to identify transactions that are unlikely to require additional identity verification and examine how machine learning algorithms may intelligently determine the appropriate occasions for bypassing further authentication checks.
        - To determine the potential benefits that may accrue to both merchants and consumers, which includes possible rises in satisfaction levels and sales conversion rates.
    """)

elif selected == "Dataset Information":
    st.subheader(f"**{selected}**")

    # Dropdown menu for selecting the dataset
    dataset_option = st.selectbox(
        "Select a dataset:",
        options=["Original Dataset", "Feature Engineered Dataset"]
    )

    # Determine the dataset based on user selection
    if dataset_option == "Original Dataset":
        data = scadata
        data_key = 'scadata'
        st.write("You selected the **Original Dataset**.")
    else:
        data = cleaned_data
        data_key = 'cleaned_data'
        st.write("You selected the **Feature Engineered Dataset**.")

    # Checkboxes to select which information to display
    show_data_dict = st.checkbox("Data Dictionary")
    show_describe = st.checkbox("Describe")
    show_data_types = st.checkbox("Data Types")
    show_missing_values = st.checkbox("Missing Values")
    show_correlation = st.checkbox("Correlation")

    # Display selected information
    if show_data_dict:
        st.subheader("Data Dictionary")
        dictionary = data_dictionary.get(data_key, {})
        for col, desc in dictionary.items():
            st.write(f"**{col}:** {desc}")

    if show_describe:
        st.subheader("Describe")
        st.write(data.describe())

    if show_data_types:
        st.subheader("Data Types")
        st.write(data.dtypes)

    if show_missing_values:
        st.subheader("Missing Values")
        st.write(data.isna().sum())

    if show_correlation:
        st.subheader("Correlation")
        st.write(data.corr())

elif selected == "Training Results":
    st.subheader(f"**Training Results**")

    # First dropdown for selecting the main model
    model_option = st.selectbox(
        "Select a model to view results:",
        options=["Logistic Regression", "Random Forest", "Support Vector Classifier", "Autoencoder/Multilayer Perceptron"]
    )

    # Display table for the selected main model
    if model_option == "Logistic Regression":
        st.write("Results for Logistic Regression model:")
        data = {
            "Model": ["logreg1", "logreg1", "logreg2", "logreg2", "logreg3", "logreg3", "logreg4", "logreg4"],
            "AUC": [0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.85, 0.85],
            "Class": [0, 1, 0, 1, 0, 1, 0, 1],
            "Precision": [0.24, 0.84, 0.43, 0.84, 0.42, 0.97, 0.43, 0.98],
            "Recall": [0.10, 0.94, 0.01, 1.0, 0.90, 0.76, 0.90, 0.76],
            "F1 Score": [0.14, 0.89, 0.01, 0.91, 0.57, 0.85, 0.58, 0.86],
            "Confusion Matrix": [
                "[[  85  387], [ 155 2297]]", "[[  85  387], [ 155 2297]]",
                "[[   3  469], [   4 2448]]", "[[   3  469], [   4 2448]]",
                "[[ 426   46], [ 590 1862]]", "[[ 426   46], [ 590 1862]]",
                "[[ 424   48], [ 574 1878]]", "[[ 424   48], [ 574 1878]]"
            ]
        }
        df = pd.DataFrame(data)
        st.table(df)

    elif model_option == "Random Forest":
        st.write("Results for Random Forest model:")
        data = {
            "Model": ["rf1", "rf1", "rf2", "rf2", "rf3", "rf3", "rf4", "rf4", "rf5", "rf5"],
            "AUC": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Class": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Precision": [1, 1, 0.98, 1, 0.99, 1, 0.99, 1, 0.99, 1],
            "Recall": [0.99, 1, 0.99, 1, 1, 1, 1, 1, 0.99, 1],
            "F1 Score": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Confusion Matrix": [
                "[[ 468    4], [   0 2452]]", "[[ 468    4], [   0 2452]]",
                "[[ 314    3], [   0 1632]]", "[[ 314    3], [   0 1632]]",
                "[[ 468    4], [   0 2452]]", "[[ 468    4], [   0 2452]]",
                "[[ 468    4], [   0 2452]]", "[[ 468    4], [   0 2452]]",
                "[[ 314    3], [   0 1632]]", "[[ 314    3], [   0 1632]]"
            ]
        }
        df = pd.DataFrame(data)
        st.table(df)

    elif model_option == "Support Vector Classifier":
        st.write("Results for Support Vector Classifier model:")
        data = {
            "Model": ["scv01", "scv01", "scv02", "scv02", "scv03", "scv03", "scv04", "scv04"],
            "AUC": [0.47, 0.47, 0.82, 0.82, 0.72, 0.72, 0.76, 0.76],
            "Class": [0, 1, 0, 1, 0, 1, 0, 1],
            "Precision": [0.71, 0.84, 0.40, 0.98, 0.43, 0.98, 0.79, 0.90],
            "Recall": [0.01, 1, 0.91, 0.74, 0.92, 0.77, 0.46, 0.98],
            "F1 Score": [0.02, 0.91, 0.55, 0.84, 0.59, 0.86, 0.56, 0.94],
            "Confusion Matrix": [
                "[[   5  467], [   2 2450]]", "[[   5  467], [   2 2450]]",
                "[[   3  314], [   2 1630]]", "[[   3  314], [   2 1630]]",
                "[[ 436   36], [ 567 1885]]", "[[ 436   36], [ 567 1885]]",
                "[[ 209  263], [  60 2392]]", "[[ 209  263], [  60 2392]]"
            ]
        }
        df = pd.DataFrame(data)
        st.table(df)

    elif model_option == "Autoencoder/Multilayer Perceptron":
        st.write("Results for Autoencoder/Multilayer Perceptron model:")
        data = {
            "Model": ["model01", "model01", "model02", "model02", "mlp01", "mlp01", "mlp02", "mlp02"],
            "AUC": [0.66, 0.66, 0.65, 0.65, 0.93, 0.93, 0.76, 0.76],
            "Class": [0, 1, 0, 1, 0, 1, 0, 1],
            "Precision": [0.17, 0.93, 0.17, 0.96, 0.76, 0.91, 0.71, 0.84],
            "Recall": [0.98, 0.06, 0.99, 0.06, 0.50, 0.97, 0.01, 1],
            "F1 Score": [0.28, 0.11, 0.29, 0.11, 0.60, 0.94, 0.02, 0.91],
            "Confusion Matrix": [
                "[[ 2316  11], [  461 136]]", "[[ 2316  11], [  461 136]]",
                "[[ 2311   6], [  466 141]]", "[[ 2311   6], [  466 141]]",
                "[[  222  250], [   89 2363]]", "[[  222  250], [   89 2363]]",
                "[[    5  713], [    5 2450]]", "[[    5  713], [    5 2450]]"
            ]
        }
        df = pd.DataFrame(data)
        st.table(df)

    # Now show the second dropdown for selecting the sub-model
    sub_models = {
        "Logistic Regression": ["logreg1", "logreg2", "logreg3", "logreg4"],
        "Random Forest": ["rf1", "rf2", "rf3", "rf4", "rf5"],
        "Support Vector Classifier": ["scv01", "scv02", "scv03", "scv04"],
        "Autoencoder/Multilayer Perceptron": ["model01", "model02", "mlp01", "mlp02"]
    }

    sub_model_descriptions = {
    "logreg1": """
        Logistic Regression Model 1: This model is tuned with default hyperparameters.
        
        ### Model Performance Summary:
        
        **Class 1 (Positive Class)**:
        - Precision: 0.84 (84% of predicted class 1 instances were correct)
        - Recall: 0.94 (94% of actual class 1 instances were correctly identified)
        - F1-Score: 0.89 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)
        
        **Class 0 (Negative Class)**:
        - Precision: 0.24 (Only 24% of predicted class 0 instances were correct)
        - Recall: 0.10 (Only 10% of actual class 0 instances were correctly identified)
        - F1-Score: 0.14 (Poor overall performance)
        - Support: 472 (Total number of actual class 0 instances)
        
        The model has high precision and recall for class 1, reflected in its strong F1-Score of 0.89. However, it performs poorly on class 0 with low precision, recall, and F1-Score. This indicates that while the model is effective at identifying positive instances, it struggles with negative ones. To address this imbalance, methods such as resampling, adjusting class weights, or using different algorithms could be considered.
        
        - **True Positives (TP)**: 2297
        - **True Negatives (TN)**: 85
        - **False Positives (FP)**: 387
        - **False Negatives (FN)**: 155
    """,
    
    "logreg2": """
        Logistic Regression Model 2: In this phase, we applied **StandardScaler** to standardize the features, transforming them to have a mean of zero and a standard deviation of one. This helps to ensure that all input features contribute equally to the model's learning process. We also applied L2 regularization by setting the regularization parameter (𝐶) to 0.01. This penalizes large coefficients, promoting a more stable and generalizable model.
        
        ### Model Performance Summary:
        
        **Class 0 (Negative Class)**:
        - Precision: 0.43 (43% of predicted class 0 instances were correct)
        - Recall: 0.01 (Only 1% of actual class 0 instances were correctly identified)
        - F1-Score: 0.01 (Poor overall performance)
        - Support: 472 (Total number of actual class 0 instances)
        
        **Class 1 (Positive Class)**:
        - Precision: 0.84 (84% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 0.91 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)
        
        The model performs exceptionally well for class 1, with high precision, recall, and F1-score, but it struggles significantly with class 0. Further model tuning or techniques such as resampling or adjusting class weights might be needed to improve performance on the minority class.
        
        - **True Positives (TP)**: 2448
        - **True Negatives (TN)**: 3
        - **False Positives (FP)**: 469
        - **False Negatives (FN)**: 4
    """,
    
    "logreg3": """
        Logistic Regression Model 3: In this phase of model development, given the imbalanced nature of the dataset for the target variable, the Synthetic Minority Over-sampling Technique (SMOTE) was utilized to address this imbalance. SMOTE works by generating synthetic samples for the minority class, effectively balancing the class distribution. This approach helps the Logistic Regression model to better learn and generalize from both classes, improving its ability to predict the minority class, which is crucial for maintaining high precision and recall in an imbalanced dataset.

        By incorporating SMOTE, the model is better equipped to avoid bias towards the majority class, thereby enhancing its overall performance and reliability in predicting outcomes.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.42 (42% of predicted class 0 instances were correct)
        - Recall: 0.090 (90% of actual class 0 instances were correctly identified)
        - F1-Score: 0.57 (Average overall performance)
        - Support: 472 (Total number of actual class 0 instances)
        
        **Class 1 (Positive Class)**:
        - Precision: 0.97 (97% of predicted class 1 instances were correct)
        - Recall: 0.90 (90% of actual class 1 instances were correctly identified)
        - F1-Score: 0.57 (Average overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 1862
        - True Negatives (TN): 426
        - False Positives (FP): 46
        - False Negatives (FN): 590

        
    """,
    
    "logreg4": """
        Logistic Regression Model 4: In this iteration of our logistic regression model, we focused on enhancing the model's performance through hyperparameter tuning. Hyperparameter tuning is a crucial step in the modeling process, as it involves adjusting the parameters that govern the learning process of the model to achieve optimal performance. By fine-tuning these parameters, we aim to improve the model's ability to correctly classify instances, particularly in the presence of class imbalance.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.43 (43% of predicted class 0 instances were correct)
        - Recall: 0.90 (90% of actual class 0 instances were correctly identified)
        - F1-Score: 0.58 (Moderate overall performance)
        - Support: 472 (Total number of actual class 0 instances)
        
        **Class 1 (Positive Class)**:
        - Precision: 0.98 (98% of predicted class 1 instances were correct)
        - Recall: 0.76 (76% of actual class 1 instances were correctly identified)
        - F1-Score: 0.86 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 1878
        - True Negatives (TN): 424
        - False Positives (FP): 48
        - False Negatives (FN): 574

    """,

    "rf1": """
        Random Forest Model 1: In this initial phase of our project, we will implement a vanilla version of the Random Forest model as our foundational approach. By focusing on a straightforward implementation, we develop a basic Random Forest model without incorporating any advanced techniques, optimizations, or hyperparameter tuning. This will allow us to establish a clear baseline for performance and gain a fundamental understanding of how Random Forest operates on our data. 

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 1.00 (100% of predicted class 0 instances were correct)
        - Recall: 0.99 (99% of actual class 0 instances were correctly identified)
        - F1-Score: 1.00 (Strong overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 1.00 (100% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 1.00 (Perfect performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 2452
        - True Negatives (TN): 468
        - False Positives (FP): 4
        - False Negatives (FN): 0

    """,
    
    "rf2": """
        Random Forest Model 2: In this second iteration of the Random Forest model, we introduce key improvements, including an increase in the number of trees to 1,000, out-of-bag (OOB) scoring, and limitations on the complexity of individual trees using a maximum leaf node constraint. These changes aim to enhance model accuracy and computational efficiency.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.98 (98% of predicted class 0 instances were correct)
        - Recall: 0.99 (90% of actual class 0 instances were correctly identified)
        - F1-Score: 1.0 (Strong overall performance)
        - Support: 317 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 1.00 (100% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 1.00 (Perfect performance)
        - Support: 1632 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 1632
        - True Negatives (TN): 314
        - False Positives (FP): 3
        - False Negatives (FN): 0

    """,
    
    "rf3": """
        Random Forest Model 3: In this iteration, we incorporate feature importance analysis to understand which variables most influence the model's predictions. By quantifying the contribution of each feature, we gain insights into the drivers behind the approval or decline decisions, which can further refine the model and improve interpretability.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.99 (99% of predicted class 0 instances were correct)
        - Recall: 1.0 (100% of actual class 0 instances were correctly identified)
        - F1-Score: 1.0 (Strong overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 1.00 (100% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 1.00 (Perfect performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 2452
        - True Negatives (TN): 468
        - False Positives (FP): 4
        - False Negatives (FN): 0


    """,
    
    "rf4": """
        Random Forest Model 4: In this version, feature scaling is introduced as a key preprocessing step to standardize the range of independent variables. This adjustment ensures that no single feature dominates the learning process due to its scale, thereby improving model balance and predictive performance.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.99 (99% of predicted class 0 instances were correct)
        - Recall: 1.00 (100% of actual class 0 instances were correctly identified)
        - F1-Score: 1.0 (Strong overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 1.00 (100% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 1.00 (Perfect performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 2452
        - True Negatives (TN): 468
        - False Positives (FP): 4
        - False Negatives (FN): 0

    """,
    
    "rf5": """
        Random Forest Model 5: In the final iteration, we introduce hyperparameter tuning using GridSearchCV to optimize key parameters such as `n_estimators`, `max_depth`, and `min_samples_split`. This approach helps identify the best model configuration for improved accuracy and robustness.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.99 (99% of predicted class 0 instances were correct)
        - Recall: 0.99 (99% of actual class 0 instances were correctly identified)
        - F1-Score: 1.00 (Strong overall performance)
        - Support: 317 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 1.00 (100% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 1.00 (Perfect performance)
        - Support: 1632 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 1632
        - True Negatives (TN): 314
        - False Positives (FP): 3
        - False Negatives (FN): 0

    """,

     "scv01": """
        SVC Model 01: In this initial phase of our project, we implement a baseline version of the Support Vector Classifier (SVC). This model is built using default parameters and serves as a reference point for evaluating more advanced versions of SVC. By setting up this straightforward version, we aim to gain a clear understanding of how the basic SVC operates and performs on our dataset, providing a foundation for further optimization.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.71 (71% of predicted class 0 instances were correct)
        - Recall: 0.01 (1% of actual class 0 instances were correctly identified)
        - F1-Score: 0.02 (Low overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.84 (84% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 0.91 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 2450
        - True Negatives (TN): 5
        - False Positives (FP): 467
        - False Negatives (FN): 2


    """,

    "scv02": """
        SVC Model 02: To improve performance for class 0, we apply **SMOTE (Synthetic Minority Over-sampling Technique)** in this iteration. SMOTE generates synthetic samples for the minority class, aiming to balance the class distribution and improve model performance, particularly for class 0.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.40 (40% of predicted class 0 instances were correct)
        - Recall: 0.91 (91% of actual class 0 instances were correctly identified)
        - F1-Score: 0.55 (Average overall performance)
        - Support: 317 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.98 (98% of predicted class 1 instances were correct)
        - Recall: 0.74 (74% of actual class 1 instances were correctly identified)
        - F1-Score: 0.84 (Strong overall performance)
        - Support: 1632 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 1630
        - True Negatives (TN): 3
        - False Positives (FP): 314
        - False Negatives (FN): 2


    """,

    "scv03": """
        SVC Model 03: In this iteration, we focus on optimizing the SVC model by tuning key hyperparameters such as the kernel, regularization parameter (C), and gamma. The goal is to improve the model's generalization ability and enhance its performance on unseen data.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.43 (43% of predicted class 0 instances were correct)
        - Recall: 0.92 (92% of actual class 0 instances were correctly identified)
        - F1-Score: 0.59 (Moderate overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.98 (98% of predicted class 1 instances were correct)
        - Recall: 0.77 (77% of actual class 1 instances were correctly identified)
        - F1-Score: 0.86 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 1885
        - True Negatives (TN): 436
        - False Positives (FP): 36
        - False Negatives (FN): 567


    """,

    "scv04": """
        SVC Model 04: In this final iteration, we conduct comprehensive hyperparameter tuning by exploring different kernels, including linear, radial basis function (RBF), and polynomial. We also adjust parameters such as the regularization parameter (C) and gamma to optimize performance.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.79 (79% of predicted class 0 instances were correct)
        - Recall: 0.46 (46% of actual class 0 instances were correctly identified)
        - F1-Score: 0.56 (Moderate overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.90 (90% of predicted class 1 instances were correct)
        - Recall: 0.98 (98% of actual class 1 instances were correctly identified)
        - F1-Score: 0.94 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 2392
        - True Negatives (TN): 209
        - False Positives (FP): 263
        - False Negatives (FN): 60

    """,
     "model01": """
        Autoencoder Model 01: This is the first iteration of an autoencoder model using TensorFlow and Keras. The data is preprocessed by applying one-hot encoding and scaling the features. The autoencoder consists of an encoder that compresses the input data and a decoder that reconstructs it. The model is trained to minimize reconstruction error, learning the data patterns. After training, reconstruction errors are computed on the test set.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.17 (Only 17% of predicted class 0 instances were correct)
        - Recall: 0.98 (98% of actual class 0 instances were correctly identified)
        - F1-Score: 0.28 (Low overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.93 (93% of predicted class 1 instances were correct)
        - Recall: 0.06 (Only 6% of actual class 1 instances were correctly identified)
        - F1-Score: 0.11 (Poor overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 136
        - True Negatives (TN): 461
        - False Positives (FP): 11
        - False Negatives (FN): 2316


    """,

    "model02": """
        Autoencoder Model 02: In this iteration, SMOTE (Synthetic Minority Over-sampling Technique) is introduced to address class imbalance in the dataset. SMOTE generates synthetic samples for the minority class, ensuring a more balanced training set to help improve the autoencoder's classification performance.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.17 (Only 17% of predicted class 0 instances were correct)
        - Recall: 0.99 (99% of actual class 0 instances were correctly identified)
        - F1-Score: 0.29 (Moderate performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.96 (96% of predicted class 1 instances were correct)
        - Recall: 0.06 (Only 5% of actual class 1 instances were correctly identified)
        - F1-Score: 0.11 (Poor overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 141
        - True Negatives (TN): 466
        - False Positives (FP): 6
        - False Negatives (FN): 2311


    """,

    "mlp01": """
        MLP Model 01: In this third iteration, a Multi-Layer Perceptron (MLP) classifier is introduced. The autoencoder reduces the dimensionality of the input data, extracting key features, while the MLP performs the final classification. The data is preprocessed with one-hot encoding, feature scaling, and the autoencoder's encoder is used to transform the data into a compact representation for better classification.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.76 (76% of predicted class 0 instances were correct)
        - Recall: 0.50 (50% of actual class 0 instances were correctly identified)
        - F1-Score: 0.60 (Moderate overall performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.91 (91% of predicted class 1 instances were correct)
        - Recall: 0.97 (97% of actual class 1 instances were correctly identified)
        - F1-Score: 0.94 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 2363
        - True Negatives (TN): 222
        - False Positives (FP): 250
        - False Negatives (FN): 89


    """,

    "mlp02": """
        MLP Model 02: In this iteration, the MLP model is optimized using GridSearchCV to tune hyperparameters such as hidden layer sizes, activation functions, and solvers. Preprocessing steps include one-hot encoding, feature scaling, and dimensionality reduction via PCA. This version aims to improve robustness and handle class imbalance more effectively.

        ### Model Performance Summary:

        **Class 0 (Negative Class)**:
        - Precision: 0.71 (71% of predicted class 0 instances were correct)
        - Recall: 0.01 (Only 1% of actual class 0 instances were correctly identified)
        - F1-Score: 0.02 (Very poor performance)
        - Support: 472 (Total number of actual class 0 instances)

        **Class 1 (Positive Class)**:
        - Precision: 0.84 (84% of predicted class 1 instances were correct)
        - Recall: 1.00 (100% of actual class 1 instances were correctly identified)
        - F1-Score: 0.91 (Strong overall performance)
        - Support: 2452 (Total number of actual class 1 instances)

        **Confusion Matrix:**
        - True Positives (TP): 2450
        - True Negatives (TN): 5
        - False Positives (FP): 713
        - False Negatives (FN): 5


    """
    
}

    # Dynamically show second dropdown based on first model selection
    if model_option in sub_models:
        sub_model_option = st.selectbox(
            f"Select a specific {model_option} model:",
            options=sub_models[model_option]
        )

        # Display the description for the selected sub-model
        st.write(sub_model_descriptions.get(sub_model_option, "No description available for this model."))

elif selected == "Models":
    st.subheader(f"**Compare Models**")

    # Spider chart data for the best models
    models_data = {
        "Logistic Regression (lg4)": {
            "AUC": 0.85, "F1 Score": 0.86, "Recall": 0.77, "Precision": 0.98
        },
        "Random Forest (rf5)": {
            "AUC": 1.0, "F1 Score": 1.0, "Recall": 1.0, "Precision": 1.0
        },
        "Support Vector Classifier (svc4)": {
            "AUC": 0.76, "F1 Score": 0.95, "Recall": 0.97, "Precision": 0.92
        },
        "Multilayer Perceptron (mlp01)": {
            "AUC": 0.93, "F1 Score": 0.94, "Recall": 0.97, "Precision": 0.91
        }
    }

    # Checkboxes to select which models to compare
    lg_checkbox = st.checkbox("Logistic Regression (lg4)")
    rf_checkbox = st.checkbox("Random Forest (rf5)")
    svc_checkbox = st.checkbox("Support Vector Classifier (svc4)")
    mlp_checkbox = st.checkbox("Multilayer Perceptron (mlp01)")

    # Collect selected models
    selected_models = []
    if lg_checkbox:
        selected_models.append("Logistic Regression (lg4)")
    if rf_checkbox:
        selected_models.append("Random Forest (rf5)")
    if svc_checkbox:
        selected_models.append("Support Vector Classifier (svc4)")
    if mlp_checkbox:
        selected_models.append("Multilayer Perceptron (mlp01)")

    if selected_models:
        # Define categories for the spider chart
        categories = ["AUC", "F1 Score", "Recall", "Precision"]
        N = len(categories)

        # Function to create a spider chart
        def create_spider_chart(models):
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Plot each model
            for model in models:
                values = list(models_data[model].values())
                values += values[:1]
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.25)
                
                # Display values on the chart (move them further)
                for i, value in enumerate(values[:-1]):
                    angle_rad = angles[i]
                    ax.text(angle_rad, value + 0.1, f'{value:.2f}', horizontalalignment='center', size=10, color='black')

            # Add labels and title
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_yticklabels([])
            
            # Adjust legend size and position
            plt.legend(loc='upper left', bbox_to_anchor=(1.3, 1), fontsize='small', frameon=False)
            plt.title("Model Comparison (Spider Chart)", size=14, color="#333", y=1.1)
            st.pyplot(fig)

        # Create and display the spider chart with selected models
        create_spider_chart(selected_models)
