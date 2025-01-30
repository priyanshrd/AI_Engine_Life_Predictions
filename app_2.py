import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF

# Load the trained model
model = load_model("fleet_maintenance_model.h5")

# Set page layout
st.set_page_config(page_title="Fleet Maintenance Dashboard", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; font-family: 'Arial', sans-serif; }
    .stButton > button { background-color: #007bff; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("📂 Upload Fleet Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Ensure a file is uploaded
if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # 📋 **View Uploaded Data**
    with st.expander("📋 View Uploaded Data", expanded=True):
        st.write("### Uploaded Data Preview")
        st.dataframe(raw_data)

    # Extract necessary features
    feature_columns = ['Engine RPM', 'Lube oil pressure', 'Fuel pressure', 
                       'Coolant pressure', 'Lube oil temperature', 'Coolant temperature']
    features = raw_data[feature_columns]

    # Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Predict vehicle condition
    predictions = model.predict(features_scaled)
    raw_data['Predicted Condition'] = np.where(predictions > 0.5, 'Healthy', 'At Risk')

    # 📈 **Prediction Results with Conditional Highlighting**
    # 📈 Highlight full row in dark red for "At Risk" vehicles
    def highlight_risk(row):
        """Apply dark red background with white text for 'At Risk' rows."""
        color = '#b30000' if row['Predicted Condition'] == 'At Risk' else 'white'
        text_color = 'white' if row['Predicted Condition'] == 'At Risk' else 'black'
        return [f'background-color: {color}; color: {text_color}'] * len(row)

    # 📋 **Prediction Results Table with Highlighting**
    with st.expander("🔍 Prediction Results", expanded=True):
        st.write("### AI-Powered Predictions")
        
        # Apply row-wise styling
        styled_df = raw_data.style.apply(highlight_risk, axis=1)
        
        # Display the styled DataFrame
        st.dataframe(styled_df)


    # Fleet health summary
    healthy_count = (raw_data['Predicted Condition'] == 'Healthy').sum()
    at_risk_count = (raw_data['Predicted Condition'] == 'At Risk').sum()

    # 🥧 **Pie Chart - Fleet Health Distribution**
    with st.expander("📊 Fleet Health Overview", expanded=True):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie([healthy_count, at_risk_count], labels=['Healthy', 'At Risk'], 
               autopct='%1.1f%%', colors=['#28a745', '#dc3545'], startangle=90)
        ax.set_title("Fleet Health Distribution")
        st.pyplot(fig)

    # 📊 **Interactive Bar Chart**
    health_summary = raw_data['Predicted Condition'].value_counts().reset_index()
    health_summary.columns = ['Condition', 'Count']
    
    fig_bar = px.bar(
        health_summary,
        x='Condition',
        y='Count',
        color='Condition',
        color_discrete_map={'Healthy': '#28a745', 'At Risk': '#dc3545'},
        title="Fleet Health Status",
        template='plotly_white'
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(title_x=0.5, xaxis_title="Condition", yaxis_title="Count")
    st.plotly_chart(fig_bar)

    # 🚨 **At-Risk Vehicle Analysis**
    at_risk_data = raw_data[raw_data['Predicted Condition'] == 'At Risk']

    if not at_risk_data.empty:
        # Compute average values for at-risk vehicles
        at_risk_avg = at_risk_data[feature_columns].mean().reset_index()
        at_risk_avg.columns = ['Feature', 'Average Value']

        fig_risk = px.bar(
            at_risk_avg, x='Average Value', y='Feature', orientation='h', 
            title="Average Feature Metrics (At Risk Vehicles)",
            color='Feature', template="plotly_white", text='Average Value'
        )
        fig_risk.update_layout(title_x=0.5, xaxis_title="Average Value", yaxis_title="Feature")
        st.plotly_chart(fig_risk)

    # 📌 **Scatter Plot for RPM vs Coolant Temperature**
    fig_scatter = px.scatter(
        raw_data, x='Engine rpm', y='Coolant temp', color='Predicted Condition',
        title="Engine RPM vs Coolant Temperature",
        color_discrete_map={'Healthy': '#28a745', 'At Risk': '#dc3545'},
        template="plotly_white"
    )
    fig_scatter.update_layout(title_x=0.5, xaxis_title="Engine RPM", yaxis_title="Coolant Temperature")
    st.plotly_chart(fig_scatter)

    # 📄 **Generate PDF Report**
    def generate_pdf(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 0, 0)

        # Title
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="Fleet Maintenance Report", ln=True, align='C')
        pdf.ln(10)

        # Summary
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Total Vehicles: {len(data)}", ln=True)
        pdf.cell(200, 10, txt=f"Healthy Vehicles: {healthy_count}", ln=True)
        pdf.cell(200, 10, txt=f"At-Risk Vehicles: {at_risk_count}", ln=True)
        pdf.ln(10)

        # Table Header
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(40, 10, "Engine rpm", border=1)
        pdf.cell(40, 10, "Lub Oil Pressure", border=1)
        pdf.cell(40, 10, "Coolant Temp", border=1)
        pdf.cell(40, 10, "Condition", border=1)
        pdf.ln()

        # Table Data
        pdf.set_font("Arial", size=10)
        for _, row in data.iterrows():
            pdf.cell(40, 10, f"{row['Engine rpm']:.2f}", border=1)
            pdf.cell(40, 10, f"{row['Lub oil pressure']:.2f}", border=1)
            pdf.cell(40, 10, f"{row['Coolant temp']:.2f}", border=1)

            # Highlight "At Risk" rows
            if row['Predicted Condition'] == "At Risk":
                pdf.set_text_color(255, 0, 0)  # Red
            pdf.cell(40, 10, row['Predicted Condition'], border=1)
            pdf.set_text_color(0, 0, 0)  # Reset to black
            pdf.ln()

        # Save PDF
        pdf_path = "fleet_report.pdf"
        pdf.output(pdf_path)
        return pdf_path

    # 📥 **Download Report Button**
    if st.button("📥 Download Fleet Report"):
        pdf_file = generate_pdf(raw_data)
        with open(pdf_file, "rb") as f:
            st.download_button("Download Report", f, file_name="fleet_report.pdf")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
