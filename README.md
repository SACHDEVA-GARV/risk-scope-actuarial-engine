# RiskScope: An Actuarial Pricing Engine

**Live Demo:** [**<-- MY STREAMLIT CLOUD URL HERE**]([https://my-app-url.streamlit.app/](https://risk-scope-actuarial-engine.streamlit.app/))

---

### Project Overview

RiskScope is a comprehensive web application designed to simulate the core activities of an Insurance Consulting Analyst Engineer. This tool provides an interactive dashboard for analyzing insurance risk factors, an automated data profiling report, and a predictive engine for estimating premium charges. The application is built entirely in Python, leveraging a modern data science stack to deliver a robust, end-to-end solution that showcases technical and mathematical thinking.

This project was specifically developed to align with the skills and responsibilities outlined for the **Insurance Consulting Analyst Engineer** role at **WTW**, demonstrating a proactive approach to building automated solutions for pricing and risk analysis.



---

### Key Features

This application is built with a multi-tab interface to provide a comprehensive user experience:

1.  **📊 Curated Dashboard:**
    * **Dynamic Filtering:** Users can filter the entire dataset by region, age range, and gender.
    * **Key Performance Indicators:** High-level metrics (Total Clients, Average Age, Average Charge) that update in real-time.
    * **Interactive Visualizations:** A suite of Plotly charts analyzing the impact of core risk factors like smoking status, BMI, and age on insurance premiums.
    * **Data Export:** A download button to export the filtered data as a CSV file, simulating the creation of a client deliverable.

2.  **🤖 Automated Analysis:**
    * **Automated Reporting:** Integrates `ydata-profiling` to generate a deep, automated exploratory data analysis (EDA) report on the filtered dataset.
    * **In-Depth Insights:** Provides detailed analysis of variables, correlations, missing values, and more without any manual coding.

3.  **🔮 Premium Predictor:**
    * **Predictive Modeling:** Utilizes a Linear Regression model trained with Scikit-learn to predict insurance premiums based on user inputs.
    * **Model Validation:** Transparently displays key model performance metrics (**R² Score** and **RMSE**) to demonstrate a critical understanding of model reliability.
    * **Explainable AI (XAI):** Implements **SHAP (SHapley Additive exPlanations)** to generate a force plot for each prediction. This feature explains *why* the model made a certain prediction by showing the positive and negative impact of each input feature.

---

### Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas
* **Data Visualization:** Plotly Express
* **Automated EDA:** ydata-profiling
* **Machine Learning:** Scikit-learn
* **Model Explainability:** SHAP

---

### Setup and Running Locally

To run this application on your local machine, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/SACHDEVA-GARV/risk-scope-actuarial-engine.git](https://github.com/SACHDEVA-GARV/risk-scope-actuarial-engine.git)
    cd risk-scope-actuarial-engine
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    The required libraries are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
The application will open in your default web browser.
