import streamlit as st
import pandas as pd
import plotly.express as px

# Profiling and HTML rendering
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

# Model training utilities
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Evaluation metrics and explainability
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import shap

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="RiskScope: An Actuarial Pricing Engine", 
    layout="wide"
)

# ------------------ Data Loading ------------------
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

df = load_data()

# ------------------ Model Training ------------------
@st.cache_resource
def train_model(df):
    """Prepare data, train regression model, evaluate metrics, and build SHAP explainer."""
    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    target = 'charges'

    X = df[features]
    y = df[target]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess categorical variables
    categorical_features = ['sex', 'smoker', 'region']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    # Build regression pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train and evaluate on test split
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Retrain on full dataset for final use
    model_pipeline.fit(X, y)

    # Build SHAP explainer
    explainer = shap.Explainer(
        model_pipeline.named_steps['regressor'],
        model_pipeline.named_steps['preprocessor'].transform(X)
    )

    return model_pipeline, r2, rmse, explainer

model, r2, rmse, explainer = train_model(df)

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Dashboard Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df['region'].unique(),
    default=df['region'].unique()
)

age_selection = st.sidebar.slider(
    'Select Age Range',
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

sex_selection = st.sidebar.radio(
    "Select Gender",
    options=['All', 'male', 'female'],
    horizontal=True
)

# Apply filters
mask = df['region'].isin(region) & df['age'].between(age_selection[0], age_selection[1])
if sex_selection != 'All':
    mask &= df['sex'] == sex_selection

df_filtered = df[mask]

# ------------------ Main Page ------------------
st.title("📈 RiskScope: An Actuarial Pricing Engine")
st.markdown("---")

# Tab structure
tab1, tab2, tab3 = st.tabs(["📊 Curated Dashboard", "🤖 Automated Analysis", "🔮 Premium Predictor"])

# ------------------ Tab 1: Curated Dashboard ------------------
with tab1:
    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please select at least one region.")
    else:
        # Display key summary statistics
        st.subheader("Key Metrics")
        total_clients = df_filtered.shape[0]
        average_age = round(df_filtered['age'].mean())
        average_charges = int(df_filtered['charges'].mean())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clients", f"{total_clients}")
        col2.metric("Average Client Age", f"{average_age} years")
        col3.metric("Average Annual Charge", f"$ {average_charges:,}")
        st.markdown("---")

        # Performance guidance for the user
        st.info("💡 To ensure optimal performance, charts are loaded on-demand. Please expand the sections below to view the visualizations.")

        # Core Risk Factor Analysis Section
        with st.expander("Expand to see Core Risk Factor Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                smoker_charges = df_filtered.groupby('smoker')['charges'].mean().round(2).reset_index()
                fig_smoker = px.bar(smoker_charges, x='smoker', y='charges', title="Average Charges: Smokers vs. Non-Smokers", text='charges', color='smoker', color_discrete_map={'yes': '#FF5733', 'no': '#33C1FF'})
                st.plotly_chart(fig_smoker, use_container_width=True)
            with col2:
                fig_box = px.box(df_filtered, x='smoker', y='charges', title="Distribution of Charges by Smoker Status", color='smoker', color_discrete_map={'yes': '#FF5733', 'no': '#33C1FF'})
                st.plotly_chart(fig_box, use_container_width=True)

        # Demographic & Health Analysis Section
        with st.expander("Expand to see Demographic & Health Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter = px.scatter(df_filtered, x='bmi', y='charges', color='smoker', title="BMI vs. Charges", hover_data=['age', 'sex'])
                st.plotly_chart(fig_scatter, use_container_width=True)
            with col2:
                fig_hist = px.histogram(df_filtered, x='age', title="Distribution of Client Ages", nbins=20)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # Data Preview and Download
        with st.expander("View Filtered Data"):
            st.dataframe(df_filtered)
            
        st.download_button(
            label="📥 Download Filtered Data as CSV", 
            data=df_filtered.to_csv(index=False).encode('utf-8'), 
            file_name='filtered_insurance_data.csv', 
            mime='text/csv'
        )

# ------------------ Tab 2: Automated Analysis ------------------
with tab2:
    st.header("Automated Data Profile Report")
    st.write("Click the button below to generate a detailed profile of the filtered data. This can take a few moments on the free server.")

    # Only generate the report when the user clicks the button
    if st.button("Generate Report"):
        if not df_filtered.empty:
            # Show a spinner while the report is being generated
            with st.spinner("Generating report, please wait..."):
                pr = ProfileReport(df_filtered, explorative=True, minimal=True)
                html(pr.to_html(), height=1000, scrolling=True)
        else:
            st.warning("The report cannot be generated because no data is selected.")

# ------------------ Tab 3: Premium Predictor ------------------
with tab3:
    st.header("Predict Your Insurance Premium")
    st.write("Enter client details below to generate a premium prediction.")

    # Show model performance metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Model R² Score", f"{r2:.3f}")
    col2.metric("Model RMSE", f"${rmse:,.2f}")
    st.markdown("---")

    # Input form for prediction
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=18, max_value=100, value=30)
            bmi = st.slider("Body Mass Index (BMI)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
            children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
        with col2:
            sex_options = df['sex'].unique()
            smoker_options = df['smoker'].unique()
            region_options = df['region'].unique()
            sex = st.selectbox("Gender", options=sex_options)
            smoker = st.selectbox("Smoker", options=smoker_options)
            region = st.selectbox("Region", options=region_options)

        submitted = st.form_submit_button("Predict Premium")

        # Run prediction
        if submitted:
            input_data = pd.DataFrame({
                'age': [age], 'sex': [sex], 'bmi': [bmi],
                'children': [children], 'smoker': [smoker], 'region': [region]
            })
            prediction = model.predict(input_data)[0]
            st.metric(label="Predicted Annual Premium", value=f"${prediction:,.2f}")

            # SHAP explanation
            st.subheader("How the Prediction Was Made")
            input_processed = model.named_steps['preprocessor'].transform(input_data)
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            shap_values = explainer(input_processed)

            st.write("The plot below highlights features that increase (red) or decrease (blue) the premium prediction.")

            force_plot = shap.force_plot(
                explainer.expected_value, shap_values.values[0],
                feature_names=feature_names, matplotlib=False
            )

            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            html(shap_html, height=200, scrolling=True)
