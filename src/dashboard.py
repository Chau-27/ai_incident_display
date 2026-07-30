import os

import altair as alt
import streamlit as st
import pandas as pd
import joblib

# Set up the Streamlit page configuration
st.set_page_config(page_title="AI Risk Assessment", layout="wide")

# Load the model and its feature names
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "default_model.pkl")
try:
    model = joblib.load(model_path)
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        n_features = getattr(model, "n_features_in_", 0)
        feature_names = [f"Feature_{i + 1}" for i in range(n_features)]
    n_features = len(feature_names)
except Exception as e:
    st.error(f"Error loading model: {e}")
    feature_names = []
    n_features = 0

# Define the input format for each feature here.
# Allowed input types:
# - "binary": renders a 0/1 select box
# - "continuous": renders a numeric input with the configured step
#
# Example:
# FEATURE_INPUT_FORMATS = {
#     "physical_objects": {"type": "binary"},
#     "impact_on_critical_services": {"type": "continuous", "step": 0.5, "min": 0.0, "max": 5.0},
# }

# Default input as binary for all features, unless specified otherwise in CONTINUOUS_FEATURES below.
FEATURE_INPUT_FORMATS = {
    feature_name: {"type": "binary"}
    for feature_name in feature_names
}

# List the features that should use the shared continuous 0.5-step input.
CONTINUOUS_FEATURES = [
    "physical_objects",
    "impact_on_critical_services",
    "involving_minor",
    "detrimental_content",
    "protected_characteristic",
    "public_sector_deployment",
    "autonomy_level",
    "rights_violation"
]

for feature_name in CONTINUOUS_FEATURES:
    FEATURE_INPUT_FORMATS[feature_name] = {
        "type": "continuous",
        "step": 0.5,
        "min": 0.0,
        "max": 1.0
    }


def render_feature_input(feature_name, index):
    feature_format = FEATURE_INPUT_FORMATS.get(feature_name, {"type": "continuous", "step": 0.5})

    if feature_format.get("type") == "binary":
        return st.selectbox(
            label=feature_name,
            options=[0, 1],
            format_func=lambda value: "0 - No" if value == 0 else "1 - Yes",
            key=f"feature_{index}",
        )

    return st.number_input(
        label=feature_name,
        value=float(feature_format.get("value", 0.0)),
        step=float(feature_format.get("step", 0.5)),
        min_value=feature_format.get("min"),
        max_value=feature_format.get("max"),
        key=f"feature_{index}",
    )

# Create the input template UI
st.title("AI Risk Assessment - Data Input")

with st.form("input_form"):
    st.subheader("Enter Feature Values")
    
    # Create input fields dynamically based on model features
    input_values = []
    cols = st.columns(2)
    
    for i, feature_name in enumerate(feature_names):
        with cols[i % 2]:
            value = render_feature_input(feature_name, i)
            input_values.append(value)
    
    # Submit button
    submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        try:
            # Create a DataFrame with input values
            input_df = pd.DataFrame([input_values], columns=feature_names)

            # Calculate risk exposure for the input and compare it with the training distribution
            input_risk_exposure = float(input_df.sum(axis=1).iloc[0])
            percentile_path = os.path.join(os.path.dirname(__file__), "..", "model", "risk_percentile.csv")
            percentile_df = pd.read_csv(percentile_path)
            historical_risk_exposures = pd.to_numeric(percentile_df["risk_exposure"], errors="coerce").dropna()
            percentile_rank = float((historical_risk_exposures <= input_risk_exposure).mean() * 100)


            

            # Make prediction for the positive class (class 1)
            probabilities = model.predict_proba(input_df)[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            class_1_index = next(
                (index for index, class_name in enumerate(classes) if str(class_name) == "1"),
                1 if len(classes) > 1 else 0,
            )
            probability = float(probabilities[class_1_index])
            prediction = 1 if probability >= 0.5 else 0

            # Display results
            st.success("Prediction Complete!")
            # col1, col2 = st.columns(2)
            # with col1:
            #     st.metric("Risk Classification", f"Class {prediction}")
            # with col2:
            #     st.metric("Class 0 Confidence", f"{(1 - probability) * 100:.2f}%")
            st.title("AI Risk Assessment Result")
            st.subheader("AI Risk Estimation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated AI Incident Likelihood", f"{probability * 100:.2f}%")
            with col2:
                st.metric("Number of Open Risk Exposure", f"{input_risk_exposure:.2f}")
            with col3:
                st.metric("Open Risk Exposure Percentile", f"{percentile_rank:.2f}%")


            # Draw histogram of historical risk exposures with median and input exposure lines.
            median_risk_exposure = float(historical_risk_exposures.median())

            histogram_df = pd.DataFrame({"risk_exposure": historical_risk_exposures})
            reference_df = pd.DataFrame(
                {
                    "value": [median_risk_exposure, input_risk_exposure],
                    "label": ["Median", "Open exposure"],
                    "tooltip": [
                        f"Median ({median_risk_exposure:.2f})",
                        f"Open exposure ({input_risk_exposure:.2f}, {percentile_rank:.2f}th percentile)",
                    ],
                }
            )

            histogram_chart = (
                alt.Chart(histogram_df)
                .mark_bar(color="#4C78A8", opacity=0.45)
                .encode(
                    x=alt.X("risk_exposure:Q", bin=alt.Bin(maxbins=20), title="Risk Exposure"),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                )
            )

            reference_chart = (
                alt.Chart(reference_df)
                .mark_rule(strokeWidth=5, opacity=1)
                .encode(
                    x=alt.X("value:Q", title="Risk Exposure"),
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(domain=["Median", "Open exposure"], range=["#F58518", "#E45756"]),
                        legend=alt.Legend(title="Legend"),
                    ),
                    tooltip=[alt.Tooltip("tooltip:N", title="Reference")],
                )
            )

            histogram_chart = (
                (histogram_chart + reference_chart)
                .properties(title="Risk Exposure Distribution", height=320)
            )

            st.altair_chart(histogram_chart, use_container_width=True)

            # Display all input features with value > 0
            positive_input_features = (
                input_df.iloc[0]
                .to_frame("value")
                .reset_index()
                .rename(columns={"index": "feature"})
                .query("value > 0")
                .copy()
            )

            if not positive_input_features.empty:
                st.subheader("List of Open Risk Exposure")
                positive_input_features = positive_input_features.reset_index(drop=True)
                positive_input_features.insert(0, "No.", positive_input_features.index + 1)
                st.dataframe(
                    positive_input_features[["No.", "feature"]].rename(columns={"feature": "Feature"}),
                    hide_index=True,
                    column_config={
                        "No.": st.column_config.NumberColumn("No.", width="small", format="%d"),
                        "Feature": st.column_config.TextColumn("Feature", width="large"),
                    },
                    use_container_width=False,
                )
            else:
                st.info("No Open Risk Found.")

# Top 10 Risk Exposure
            coefficients_path = os.path.join(os.path.dirname(__file__), "..", "model", "model_coefficients.csv")
            if os.path.exists(coefficients_path):
                coefficients_df = pd.read_csv(coefficients_path)
                coefficients_df = coefficients_df[["feature", "coefficient"]].dropna().drop_duplicates(subset=["feature"])
                coefficients_df["coefficient"] = pd.to_numeric(coefficients_df["coefficient"], errors="coerce")

                positive_features = (
                    input_df.iloc[0]
                    .to_frame("value")
                    .reset_index()
                    .rename(columns={"index": "feature"})
                    .merge(coefficients_df, on="feature", how="left")
                    .query("value > 0 and coefficient.notna()")
                    .copy()
                )

                if not positive_features.empty:
                    positive_features["abs_coefficient"] = positive_features["coefficient"].abs()
                    ranked_features = (
                        positive_features.sort_values(by=["abs_coefficient", "coefficient"], ascending=[False, False])
                        .head(10)
                    )
                    st.subheader("Top 10 Open Risk / Recommended Risk Mitigation Prioritization")
                    ranked_features = ranked_features.reset_index(drop=True)
                    ranked_features.insert(0, "Rank", range(1, len(ranked_features) + 1))
                    st.dataframe(
                        ranked_features[["Rank", "feature"]].rename(
                            columns={"feature": "Feature"}
                        ),
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
                            "Feature": st.column_config.TextColumn("Feature", width="large"),
                        },
                        use_container_width=False,
                    )
                else:
                    st.info("No Open Risk Found.")
            else:
                st.info("Model file not found.")

        #     # Show probability distribution
        #     st.bar_chart(pd.DataFrame({
        #         "Probability": probabilities,
        #         "Class": [f"Class {class_name}" for class_name in classes]
        #     }).set_index("Class"))
        except Exception as e:
            st.error(f"Error making prediction: {e}")

