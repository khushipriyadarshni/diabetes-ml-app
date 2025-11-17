"""Streamlit app for diabetes risk prediction."""

# pyright: reportMissingImports=false
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.pipeline import Pipeline


# Input schema for validation
INPUT_SCHEMA = {
    "Pregnancies": {"min": 0, "max": 20, "type": "int"},
    "Glucose": {"min": 0, "max": 300, "type": "float"},
    "BloodPressure": {"min": 0, "max": 200, "type": "float"},
    "SkinThickness": {"min": 0, "max": 100, "type": "float"},
    "Insulin": {"min": 0, "max": 1000, "type": "float"},
    "BMI": {"min": 0, "max": 70, "type": "float"},
    "DiabetesPedigreeFunction": {"min": 0.0, "max": 3.0, "type": "float"},
    "Age": {"min": 0, "max": 120, "type": "int"},
}

# Columns where zero is medically invalid
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@st.cache_resource
def load_model() -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load the trained model and metadata.
    
    Returns:
        Tuple of (model, metadata) where metadata contains metrics and feature ranges
    """
    model_path = Path("models/model.joblib")
    metadata_path = Path("models/metrics.json")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    return model, metadata


def coerce_to_numeric(value: Any, feature_name: str) -> Tuple[float, Optional[str]]:
    """
    Coerce input value to numeric with clean error messages.
    
    Args:
        value: Input value to coerce
        feature_name: Name of the feature for error messages
        
    Returns:
        Tuple of (numeric_value, error_message) where error_message is None if successful
    """
    if value is None or value == "":
        return None, f"{feature_name} cannot be empty"
    
    try:
        if INPUT_SCHEMA[feature_name]["type"] == "int":
            num_value = int(float(value))
        else:
            num_value = float(value)
        
        # Check range
        schema = INPUT_SCHEMA[feature_name]
        if num_value < schema["min"] or num_value > schema["max"]:
            return (
                None,
                f"{feature_name} must be between {schema['min']} and {schema['max']}",
            )
        
        # Check for invalid zeros
        if feature_name in ZERO_INVALID_COLS and num_value == 0:
            return (
                None,
                f"{feature_name} cannot be zero (medically invalid). Please enter a valid value.",
            )
        
        # Check for negative values
        if num_value < 0:
            return None, f"{feature_name} cannot be negative"
        
        return num_value, None
    
    except (ValueError, TypeError):
        return (
            None,
            f"{feature_name} must be a valid number ({INPUT_SCHEMA[feature_name]['type']})",
        )


def validate_inputs(inputs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate all input values.
    
    Args:
        inputs: Dictionary of feature names to values
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    for feature_name, value in inputs.items():
        if feature_name not in INPUT_SCHEMA:
            continue
        
        _, error = coerce_to_numeric(value, feature_name)
        if error:
            return False, error
    
    return True, None


def create_probability_gauge(probability: float, threshold: float) -> go.Figure:
    """
    Create a probability gauge chart using Plotly.
    
    Args:
        probability: Predicted probability (0-1)
        threshold: Classification threshold (0-1)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Diabetes Risk Probability (%)"},
            delta={"reference": threshold * 100, "position": "top"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "lightgray"},
                    {"range": [threshold * 100, 100], "color": "gray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    
    fig.update_layout(height=300)
    return fig


def create_distribution_histograms(df: pd.DataFrame, numeric_cols: list) -> List[go.Figure]:
    """
    Create distribution histograms for all numeric features.
    
    Args:
        df: DataFrame with data
        numeric_cols: List of numeric column names
        
    Returns:
        List of Plotly figure objects
    """
    figures = []
    for col in numeric_cols:
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Distribution of {col}",
            labels={col: col, "count": "Frequency"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        figures.append(fig)
    return figures


def create_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> go.Figure:
    """
    Create a correlation heatmap for numeric features.
    
    Args:
        df: DataFrame with data
        numeric_cols: List of numeric column names
        
    Returns:
        Plotly figure object
    """
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        )
    )
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=500,
        margin=dict(l=100, r=20, t=60, b=100),
    )
    
    return fig


def create_boxplots(df: pd.DataFrame, numeric_cols: list) -> List[go.Figure]:
    """
    Create boxplots for all numeric features to show outliers.
    
    Args:
        df: DataFrame with data
        numeric_cols: List of numeric column names
        
    Returns:
        List of Plotly figure objects
    """
    figures = []
    for col in numeric_cols:
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                boxpoints="outliers",  # Show outliers
                marker_color="#1f77b4",
                line_color="#1f77b4",
            )
        )
        fig.update_layout(
            title=f"Boxplot of {col} (Outliers Highlighted)",
            height=400,
            yaxis_title=col,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        figures.append(fig)
    return figures


def get_feature_importance(model: Pipeline) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from RandomForest model if available.
    
    Args:
        model: Trained scikit-learn Pipeline
        
    Returns:
        DataFrame with feature names and importance scores, or None if not available
    """
    try:
        # Get the classifier from the pipeline
        classifier = model.named_steps.get("classifier")
        
        if classifier is None:
            # Try to get the last step (classifier)
            classifier = model.steps[-1][1]
        
        # Check if it's a RandomForestClassifier
        if hasattr(classifier, "feature_importances_"):
            # Get feature names from the pipeline
            feature_names = list(INPUT_SCHEMA.keys())
            
            # Create DataFrame with feature importance
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": classifier.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)
            
            return importance_df
    except Exception:
        pass
    
    return None


def create_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart for feature importance.
    
    Args:
        importance_df: DataFrame with Feature and Importance columns
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=importance_df["Importance"],
                y=importance_df["Feature"],
                orientation="h",
                marker_color="#1f77b4",
            )
        ]
    )
    
    fig.update_layout(
        title="Feature Importance from RandomForest Model",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=100, r=20, t=40, b=20),
    )
    
    return fig


def generate_recommendations(
    probability: float, prediction: int, patient_inputs: Dict[str, Any]
) -> str:
    """
    Generate personalized health recommendations based on prediction and patient data.
    
    Args:
        probability: Predicted probability of diabetes (0-1)
        prediction: Binary prediction (0 = Low Risk, 1 = High Risk)
        patient_inputs: Dictionary containing patient features (Age, BMI, Glucose, etc.)
        
    Returns:
        String containing personalized health recommendations
    """
    age = patient_inputs.get("Age", 0)
    bmi = patient_inputs.get("BMI", 0)
    glucose = patient_inputs.get("Glucose", 0)
    blood_pressure = patient_inputs.get("BloodPressure", 0)
    
    recommendations = []
    
    if prediction == 1:  # High Risk
        recommendations.append("⚠️ **High Risk Detected**: Based on your health profile, you are at elevated risk for diabetes.")
        recommendations.append("")
        recommendations.append("**Immediate Actions:**")
        
        # Glucose-specific advice
        if glucose > 140:
            recommendations.append("• **Blood Glucose Management**: Your glucose levels are elevated. Monitor your blood sugar regularly (2-3 times daily) and consult with a healthcare provider immediately.")
        elif glucose > 100:
            recommendations.append("• **Blood Glucose Monitoring**: Your glucose levels are in the pre-diabetic range. Start monitoring your blood sugar weekly and discuss with your doctor.")
        
        # BMI-specific advice
        if bmi >= 30:
            recommendations.append("• **Weight Management**: Your BMI indicates obesity. Aim to lose 5-10% of your body weight through a structured diet and exercise program. Consider consulting a nutritionist.")
        elif bmi >= 25:
            recommendations.append("• **Weight Management**: Your BMI is in the overweight range. Focus on gradual weight loss through calorie control and increased physical activity.")
        
        # Blood pressure advice
        if blood_pressure > 130:
            recommendations.append("• **Blood Pressure Control**: Your blood pressure is elevated. Reduce sodium intake, increase potassium-rich foods, and engage in regular cardiovascular exercise.")
        
        # Age-specific advice
        if age >= 45:
            recommendations.append("• **Regular Monitoring**: Given your age, schedule comprehensive health checkups every 6 months, including HbA1c and fasting glucose tests.")
        else:
            recommendations.append("• **Early Intervention**: Early detection is key. Schedule a comprehensive health checkup within the next month.")
        
        recommendations.append("")
        recommendations.append("**Lifestyle Modifications:**")
        recommendations.append("• **Diet**: Adopt a low-glycemic index diet rich in whole grains, vegetables, lean proteins, and healthy fats. Limit processed foods, sugary beverages, and refined carbohydrates.")
        recommendations.append("• **Exercise**: Engage in at least 150 minutes of moderate-intensity aerobic exercise per week (e.g., brisk walking, cycling). Include strength training 2-3 times per week.")
        recommendations.append("• **Sleep**: Aim for 7-9 hours of quality sleep per night, as poor sleep can affect glucose metabolism.")
        recommendations.append("• **Stress Management**: Practice stress-reduction techniques like meditation, yoga, or deep breathing exercises.")
        recommendations.append("")
        recommendations.append("**Medical Consultation:**")
        recommendations.append("• Schedule an appointment with your healthcare provider within 2 weeks to discuss these results and develop a personalized management plan.")
        recommendations.append("• Consider consulting an endocrinologist or diabetes specialist for comprehensive evaluation.")
        
    else:  # Low Risk
        recommendations.append("✅ **Low Risk**: Your current health profile indicates a lower risk for diabetes. Maintain these healthy habits!")
        recommendations.append("")
        recommendations.append("**Maintenance Tips:**")
        
        # Age-specific maintenance
        if age >= 45:
            recommendations.append("• **Regular Checkups**: Schedule annual comprehensive health screenings, including blood glucose and HbA1c tests, to monitor your health status.")
        else:
            recommendations.append("• **Preventive Care**: Continue with regular health checkups every 1-2 years to maintain your low-risk status.")
        
        # BMI maintenance
        if bmi >= 25:
            recommendations.append("• **Weight Maintenance**: Your BMI suggests room for improvement. Focus on maintaining a healthy weight through balanced nutrition and regular exercise.")
        else:
            recommendations.append("• **Weight Maintenance**: Maintain your healthy BMI through consistent lifestyle habits.")
        
        recommendations.append("")
        recommendations.append("**Healthy Lifestyle Practices:**")
        recommendations.append("• **Balanced Diet**: Continue consuming a well-balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins. Limit processed foods and added sugars.")
        recommendations.append("• **Regular Exercise**: Maintain at least 150 minutes of moderate-intensity exercise per week to support metabolic health.")
        recommendations.append("• **Hydration**: Drink adequate water (8-10 glasses daily) to support overall health and metabolism.")
        recommendations.append("• **Sleep Hygiene**: Maintain consistent sleep patterns with 7-9 hours of quality sleep per night.")
        recommendations.append("• **Stress Management**: Continue practicing healthy stress management techniques.")
        recommendations.append("")
        recommendations.append("**Prevention Focus:**")
        recommendations.append("• Monitor your glucose levels annually, especially if you have a family history of diabetes.")
        recommendations.append("• Stay informed about diabetes prevention strategies and maintain your current healthy lifestyle.")
    
    return "\n".join(recommendations)


def get_risk_bucket(probability: float) -> str:
    """
    Map probability to risk bucket for triage decisions.
    
    Args:
        probability: Predicted probability (0-1)
        
    Returns:
        Risk bucket: "high", "moderate", or "low"
    """
    if probability >= 0.80:
        return "high"
    elif probability >= 0.60:
        return "moderate"
    else:
        return "low"


def generate_prescription_suggestion(
    row: Dict[str, Any], prob: float, pred: int
) -> Dict[str, Any]:
    """
    Generate rule-based clinical guidance (educational only, non-diagnostic).
    
    Returns structured guidance with triage level, rationale, recommendations,
    and OTC considerations (categories only, no drugs/dosages).
    
    Args:
        row: Dictionary with patient features (Pregnancies, Glucose, BloodPressure,
             SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        prob: Predicted probability (0-1)
        pred: Binary prediction (0 = Low Risk, 1 = High Risk)
        
    Returns:
        Dictionary with keys: triage_level, rationale, recommendation_text,
        otc_considerations, see_doctor
    """
    # Extract and validate inputs with safe defaults
    glucose = float(row.get("Glucose", 0) or 0)
    blood_pressure = float(row.get("BloodPressure", 0) or 0)
    bmi = float(row.get("BMI", 0) or 0)
    age = float(row.get("Age", 0) or 0)
    
    # Initialize result structure
    result = {
        "triage_level": "routine",
        "rationale": "",
        "recommendation_text": "",
        "otc_considerations": [],
        "see_doctor": False,
    }
    
    # Rule 1: Red flags → urgent
    is_urgent = False
    urgent_reasons = []
    
    if glucose >= 200:
        is_urgent = True
        urgent_reasons.append("very high glucose")
    if blood_pressure >= 100:
        is_urgent = True
        urgent_reasons.append("elevated blood pressure")
    if age >= 65 and blood_pressure >= 90:
        is_urgent = True
        urgent_reasons.append("elevated BP in older adult")
    if bmi >= 40 and glucose >= 180:
        is_urgent = True
        urgent_reasons.append("severe obesity with high glucose")
    if prob >= 0.80:
        is_urgent = True
        urgent_reasons.append("very high risk probability")
    
    if is_urgent:
        result["triage_level"] = "urgent"
        result["see_doctor"] = True
        result["rationale"] = f"Urgent evaluation recommended due to: {', '.join(urgent_reasons[:2])}."
        result["recommendation_text"] = (
            "Immediate clinical evaluation is advised. Maintain hydration with water, "
            "follow a balanced low-glycemic diet, avoid strenuous activity until cleared, "
            "ensure adequate sleep, and practice stress management. Book an appointment with "
            "a clinician promptly; do not self-medicate. Monitor vital signs as directed."
        )
        result["otc_considerations"] = []  # No OTC for urgent cases
        return result
    
    # Rule 2: Concerning → soon
    is_concerning = False
    concerning_reasons = []
    
    if 0.60 <= prob < 0.80:
        is_concerning = True
        concerning_reasons.append("elevated risk probability")
    if 140 <= glucose < 199:
        is_concerning = True
        concerning_reasons.append("pre-diabetic glucose range")
    if 35.0 <= bmi < 39.9:
        is_concerning = True
        concerning_reasons.append("obesity class II")
    
    if is_concerning:
        result["triage_level"] = "soon"
        result["see_doctor"] = True
        result["rationale"] = f"Timely evaluation recommended due to: {', '.join(concerning_reasons[:2])}."
        result["recommendation_text"] = (
            "Schedule a clinical visit within 2-4 weeks. Focus on hydration, balanced "
            "nutrition with whole foods, regular moderate activity, quality sleep, and stress "
            "reduction. Book an appointment with a clinician; do not self-medicate. Track "
            "your health metrics weekly."
        )
        
        # OTC considerations for concerning cases
        otc_list = []
        if glucose >= 140:
            otc_list.append("sugar-free oral rehydration")
            otc_list.append("dietary fiber supplements")
        if bmi >= 35:
            otc_list.append("dietary counseling resources")
        if 90 <= blood_pressure < 100:
            otc_list.append("home BP monitoring device")
        result["otc_considerations"] = otc_list
        return result
    
    # Rule 3: Otherwise → routine
    result["triage_level"] = "routine"
    result["see_doctor"] = False
    
    # Build rationale based on risk bucket
    risk_bucket = get_risk_bucket(prob)
    if risk_bucket == "moderate":
        result["rationale"] = "Moderate risk profile suggests routine monitoring and lifestyle focus."
    else:
        result["rationale"] = "Lower risk profile indicates preventive care and healthy habits."
    
    # Build recommendation text
    base_recommendations = (
        "Maintain adequate hydration, follow a balanced diet rich in vegetables and whole grains, "
        "engage in regular physical activity, prioritize 7-9 hours of sleep nightly, and practice "
        "stress management techniques. Continue routine health checkups as recommended."
    )
    
    # Add specific guidance based on values
    specific_notes = []
    if 100 <= glucose < 140:
        specific_notes.append("Monitor glucose levels periodically.")
    if 25 <= bmi < 30:
        specific_notes.append("Consider weight management strategies.")
    if 80 <= blood_pressure < 90:
        specific_notes.append("Maintain healthy blood pressure through lifestyle.")
    
    if specific_notes:
        result["recommendation_text"] = f"{base_recommendations} {' '.join(specific_notes)}"
    else:
        result["recommendation_text"] = base_recommendations
    
    # OTC considerations for routine cases
    otc_list = []
    if 100 <= glucose < 140:
        otc_list.append("dietary fiber supplements")
    if 25 <= bmi < 30:
        otc_list.append("dietary counseling resources")
    if 80 <= blood_pressure < 90:
        otc_list.append("home BP monitoring device")
    result["otc_considerations"] = otc_list
    
    return result


def main() -> None:
    """Main Streamlit app function."""
    st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🏥", layout="wide")
    
    st.title("🏥 Diabetes Risk Prediction")
    st.markdown("Predict the risk of diabetes based on patient features")
    
    # Load model and metadata
    try:
        model, metadata = load_model()
    except FileNotFoundError as e:
        st.error(f"Model not found. Please train the model first using train.py")
        st.stop()
    
    # Sidebar for single prediction
    st.sidebar.header("Patient Information")
    
    # Get feature ranges from metadata or use defaults
    feature_ranges = metadata.get("feature_ranges", {})
    
    # Input fields
    inputs = {}
    for feature_name, schema in INPUT_SCHEMA.items():
        min_val = feature_ranges.get(feature_name, {}).get("min", schema["min"])
        max_val = feature_ranges.get(feature_name, {}).get("max", schema["max"])
        
        if schema["type"] == "int":
            inputs[feature_name] = st.sidebar.number_input(
                feature_name,
                min_value=int(schema["min"]),
                max_value=int(schema["max"]),
                value=int((min_val + max_val) / 2),
                step=1,
            )
        else:
            inputs[feature_name] = st.sidebar.number_input(
                feature_name,
                min_value=float(schema["min"]),
                max_value=float(schema["max"]),
                value=float((min_val + max_val) / 2),
                step=0.1,
            )
    
    # Threshold slider
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability threshold above which a patient is classified as having diabetes risk",
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Prediction")
        
        # Validate inputs
        is_valid, error_msg = validate_inputs(inputs)
        
        if not is_valid:
            st.error(f"❌ Input Error: {error_msg}")
            st.info("Please adjust the values in the sidebar to fix the error.")
        else:
            # Prepare input for prediction
            feature_names = list(INPUT_SCHEMA.keys())
            X_input = pd.DataFrame([inputs], columns=feature_names)
            
            # Make prediction
            try:
                probability = model.predict_proba(X_input)[0, 1]
                prediction = 1 if probability >= threshold else 0
                
                # Display results
                st.metric("Predicted Probability", f"{probability:.4f}")
                
                if prediction == 1:
                    st.warning("⚠️ **High Risk**: Patient is predicted to have diabetes risk")
                else:
                    st.success("✅ **Low Risk**: Patient is predicted to have low diabetes risk")
                
                # Probability gauge
                st.plotly_chart(create_probability_gauge(probability, threshold), use_container_width=True)
                
                # Health Recommendations
                st.markdown("---")
                st.subheader("💡 AI Health Recommendations")
                recommendations = generate_recommendations(probability, prediction, inputs)
                st.markdown(recommendations)
                
                # Clinical Guidance Section
                st.markdown("---")
                with st.expander("📝 Clinical Guidance & Prescription Suggestion (Non-diagnostic)", expanded=False):
                    # Disclaimer
                    st.warning(
                        "**⚠️ DISCLAIMER**: Educational only. Not medical advice. "
                        "No medications or dosages provided. See a licensed clinician for diagnosis and treatment."
                    )
                    
                    # Generate clinical guidance
                    try:
                        guidance = generate_prescription_suggestion(inputs, probability, prediction)
                        
                        # Triage badge with color coding
                        triage_level = guidance["triage_level"]
                        if triage_level == "urgent":
                            st.error(f"🔴 **Triage Level: URGENT**")
                        elif triage_level == "soon":
                            st.warning(f"🟠 **Triage Level: SOON**")
                        else:
                            st.success(f"🟢 **Triage Level: ROUTINE**")
                        
                        # Rationale
                        st.markdown(f"**Rationale**: {guidance['rationale']}")
                        
                        # Recommendation text
                        st.markdown("**Recommendations:**")
                        st.markdown(guidance["recommendation_text"])
                        
                        # OTC considerations
                        if guidance["otc_considerations"]:
                            st.markdown("**Educational OTC Considerations (Categories Only):**")
                            for item in guidance["otc_considerations"]:
                                st.markdown(f"• {item}")
                        
                        # See doctor indicator
                        if guidance["see_doctor"]:
                            st.info("💼 **Clinical consultation recommended**")
                    
                    except Exception as e:
                        st.error(f"Error generating clinical guidance: {str(e)}")
                        st.info("Defaulting to routine care guidance. Please consult a healthcare provider.")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    with col2:
        st.subheader("Model Information")
        
        # Display metrics
        if "metrics" in metadata:
            st.markdown("### Model Performance")
            metrics = metadata["metrics"]
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
        
        # Display feature ranges
        if feature_ranges:
            st.markdown("### Feature Ranges")
            for feature_name, ranges in feature_ranges.items():
                st.text(f"{feature_name}:")
                st.text(f"  Min: {ranges['min']:.2f}")
                st.text(f"  Max: {ranges['max']:.2f}")
    
    # Batch prediction section
    st.markdown("---")
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV file with the same columns as the training data to get predictions for multiple patients.")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="CSV file should contain columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age",
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df_batch = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = list(INPUT_SCHEMA.keys())
            missing_cols = [col for col in required_cols if col not in df_batch.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Validate and preprocess
                df_validated = df_batch[required_cols].copy()
                errors = []
                
                for idx, row in df_validated.iterrows():
                    for col in required_cols:
                        value, error = coerce_to_numeric(row[col], col)
                        if error:
                            errors.append(f"Row {idx + 1}, {col}: {error}")
                        else:
                            df_validated.at[idx, col] = value
                
                if errors:
                    st.error("Validation errors found:")
                    for error in errors[:10]:  # Show first 10 errors
                        st.text(f"  - {error}")
                    if len(errors) > 10:
                        st.text(f"  ... and {len(errors) - 10} more errors")
                else:
                    # Make predictions
                    probabilities = model.predict_proba(df_validated)[:, 1]
                    predictions = (probabilities >= threshold).astype(int)
                    
                    # Generate health recommendations for each patient
                    health_advice_list = []
                    for idx, row in df_validated.iterrows():
                        patient_data = row.to_dict()
                        advice = generate_recommendations(
                            probabilities[idx], predictions[idx], patient_data
                        )
                        # Clean up the advice for CSV (remove markdown formatting)
                        advice_clean = advice.replace("**", "").replace("• ", "- ").replace("\n", " ")
                        health_advice_list.append(advice_clean)
                    
                    # Generate clinical guidance for each patient
                    guidance_triage_list = []
                    guidance_notes_list = []
                    guidance_otc_list = []
                    
                    for idx, row in df_validated.iterrows():
                        try:
                            patient_data = row.to_dict()
                            guidance = generate_prescription_suggestion(
                                patient_data, probabilities[idx], predictions[idx]
                            )
                            guidance_triage_list.append(guidance["triage_level"])
                            guidance_notes_list.append(guidance["recommendation_text"])
                            # Join OTC considerations with semicolon
                            guidance_otc_list.append("; ".join(guidance["otc_considerations"]) if guidance["otc_considerations"] else "")
                        except Exception:
                            # Fallback to routine if error
                            guidance_triage_list.append("routine")
                            guidance_notes_list.append("Routine care recommended. Consult healthcare provider for personalized guidance.")
                            guidance_otc_list.append("")
                    
                    # Create results DataFrame
                    df_results = df_batch.copy()
                    df_results["Predicted_Probability"] = probabilities
                    df_results["Prediction"] = predictions
                    df_results["Risk_Level"] = df_results["Prediction"].map({0: "Low Risk", 1: "High Risk"})
                    df_results["Health_Advice"] = health_advice_list
                    df_results["Guidance_Triage"] = guidance_triage_list
                    df_results["Guidance_Notes"] = guidance_notes_list
                    df_results["Guidance_OTC"] = guidance_otc_list
                    
                    # Display results
                    st.success(f"✅ Processed {len(df_results)} predictions successfully!")
                    st.dataframe(df_results)
                    
                    # Display recommendations section
                    st.markdown("---")
                    st.subheader("💡 Health Recommendations Summary")
                    st.markdown("Personalized health recommendations have been generated for each patient and included in the downloadable CSV.")
                    
                    # Clinical Guidance disclaimer
                    st.info(
                        "**📝 Clinical Guidance Included**: Educational guidance columns (Guidance_Triage, Guidance_Notes, Guidance_OTC) "
                        "have been added. This is for educational purposes only and does not constitute medical advice. "
                        "No medications or dosages are provided. Consult a licensed clinician for diagnosis and treatment."
                    )
                    
                    # Show sample recommendations (first 3)
                    with st.expander("View Sample Recommendations (First 3 Patients)", expanded=False):
                        for i in range(min(3, len(df_results))):
                            st.markdown(f"**Patient {i + 1}** (Risk Level: {df_results.iloc[i]['Risk_Level']})")
                            # Get original formatted recommendations
                            original_advice = generate_recommendations(
                                probabilities[i], predictions[i], df_validated.iloc[i].to_dict()
                            )
                            st.markdown(original_advice)
                            if i < min(2, len(df_results) - 1):
                                st.markdown("---")
                    
                    # Download button
                    csv_results = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV (with Health Advice)",
                        data=csv_results,
                        file_name="diabetes_predictions.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")
    
    # Data Visualization Dashboard section
    st.markdown("---")
    st.subheader("📊 Data Visualization Dashboard")
    st.markdown("Upload a CSV file to visualize data distributions, correlations, and outliers.")
    
    # File uploader for visualization
    viz_file = st.file_uploader(
        "Choose CSV file for visualization",
        type=["csv"],
        key="viz_uploader",
        help="Upload a CSV file with the same structure as the training data to generate visualizations",
    )
    
    if viz_file is not None:
        try:
            # Read CSV
            df_viz = pd.read_csv(viz_file)
            
            # Get numeric columns (features from schema)
            numeric_cols = list(INPUT_SCHEMA.keys())
            available_cols = [col for col in numeric_cols if col in df_viz.columns]
            
            if not available_cols:
                st.warning("No matching feature columns found in the uploaded file.")
            else:
                # Filter to only available numeric columns
                df_viz_numeric = df_viz[available_cols].copy()
                
                # Remove any non-numeric values
                for col in available_cols:
                    df_viz_numeric[col] = pd.to_numeric(df_viz_numeric[col], errors="coerce")
                
                # Drop rows with all NaN
                df_viz_numeric = df_viz_numeric.dropna(how="all")
                
                if df_viz_numeric.empty:
                    st.error("No valid numeric data found in the uploaded file.")
                else:
                    st.success(f"✅ Loaded {len(df_viz_numeric)} rows with {len(available_cols)} features")
                    
                    # Tabs for different visualization types
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["📈 Distributions", "🔥 Correlation", "📦 Boxplots", "⭐ Feature Importance"]
                    )
                    
                    # Tab 1: Distribution Histograms
                    with tab1:
                        st.markdown("### Feature Distributions")
                        st.markdown("Histograms showing the distribution of each numeric feature.")
                        
                        # Display histograms in a grid (2 columns)
                        num_cols_display = 2
                        for i in range(0, len(available_cols), num_cols_display):
                            cols = st.columns(num_cols_display)
                            for j, col_name in enumerate(available_cols[i : i + num_cols_display]):
                                if j < len(cols):
                                    with cols[j]:
                                        hist_figs = create_distribution_histograms(
                                            df_viz_numeric, [col_name]
                                        )
                                        if hist_figs:
                                            st.plotly_chart(hist_figs[0], use_container_width=True)
                    
                    # Tab 2: Correlation Heatmap
                    with tab2:
                        st.markdown("### Feature Correlation Heatmap")
                        st.markdown("Shows the correlation between all numeric features. Values range from -1 (negative correlation) to +1 (positive correlation).")
                        
                        if len(available_cols) > 1:
                            corr_fig = create_correlation_heatmap(df_viz_numeric, available_cols)
                            st.plotly_chart(corr_fig, use_container_width=True)
                        else:
                            st.info("At least 2 features are required to generate a correlation heatmap.")
                    
                    # Tab 3: Boxplots
                    with tab3:
                        st.markdown("### Boxplots with Outlier Detection")
                        st.markdown("Boxplots showing the distribution and outliers for each feature. Outliers are highlighted as individual points.")
                        
                        # Display boxplots in a grid (2 columns)
                        for i in range(0, len(available_cols), num_cols_display):
                            cols = st.columns(num_cols_display)
                            for j, col_name in enumerate(available_cols[i : i + num_cols_display]):
                                if j < len(cols):
                                    with cols[j]:
                                        box_figs = create_boxplots(df_viz_numeric, [col_name])
                                        if box_figs:
                                            st.plotly_chart(box_figs[0], use_container_width=True)
                    
                    # Tab 4: Feature Importance
                    with tab4:
                        st.markdown("### Model Feature Importance")
                        st.markdown("Feature importance scores from the trained RandomForest model, showing which features are most influential in predictions.")
                        
                        importance_df = get_feature_importance(model)
                        
                        if importance_df is not None:
                            # Display importance chart
                            importance_fig = create_feature_importance_chart(importance_df)
                            st.plotly_chart(importance_fig, use_container_width=True)
                            
                            # Display importance table
                            st.markdown("#### Feature Importance Table")
                            st.dataframe(
                                importance_df.style.format({"Importance": "{:.4f}"}),
                                use_container_width=True,
                            )
                        else:
                            st.info("Feature importance is only available for RandomForest models. The current model may not support this feature.")
        
        except Exception as e:
            st.error(f"Error processing visualization file: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()

