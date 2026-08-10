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
# - "checkbox": renders a yes/no checkbox and stores 1/0
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

CHECKBOX_GROUP_KEYS = (
    "sector_of_deployment",
    "harm_distribution_basis",
    "risk_subdomain",
    "tech",
)


def default_feature_label(feature_name):
    return feature_name.replace("_", " ").title()


# Edit this mapping to control the label shown for any feature input.
# The same labels apply to binary, continuous, and checkbox-rendered features.
# Keys are model feature names, values are the display labels.
FEATURE_DISPLAY_LABELS = {
    feature_name: default_feature_label(feature_name)
    for feature_name in feature_names
}

# Example overrides:
# FEATURE_DISPLAY_LABELS["public_sector_deployment"] = "Public Sector Use"
# FEATURE_DISPLAY_LABELS["autonomy_level"] = "Autonomy Score"
# FEATURE_DISPLAY_LABELS["sector_of_deployment"] = "Sector of Deployment Enabled"
# Group headers for checkbox series can also be customized here.
FEATURE_DISPLAY_LABELS.update(
    {
        "physical_objects": "Does the AI system interact with physical objects?",
        "sector_of_deployment": "Sector of Deployment Enabled",
        "harm_distribution_basis": "Harm Distribution Basis",
        "risk_subdomain": "Risk Subdomain",
        "tech": "Tech",
    }
)

CHECKBOX_GROUP_KEYS = (
    "sector_of_deployment",
    "harm_distribution_basis",
    "risk_subdomain",
    "tech",
)


def default_feature_label(feature_name):
    return feature_name.replace("_", " ").title()


# Edit this mapping to control the label shown for any feature input.
# The same labels apply to binary, continuous, and checkbox-rendered features.
# Keys are model feature names, values are the display labels.
FEATURE_DISPLAY_LABELS = {
    feature_name: default_feature_label(feature_name)
    for feature_name in feature_names
}

# Example overrides:
# FEATURE_DISPLAY_LABELS["public_sector_deployment"] = "Public Sector Use"
# FEATURE_DISPLAY_LABELS["autonomy_level"] = "Autonomy Score"
# FEATURE_DISPLAY_LABELS["sector_of_deployment"] = "Sector of Deployment Enabled"
# Group headers for checkbox series can also be customized here.
FEATURE_DISPLAY_LABELS.update(
    {
        "tech": "What kind of technology capability does the AI system have?",    
        "harm_distribution_basis": "What kind of harm is likely to occured from the AI system usage?",
        "risk_subdomain": "What are the likely causes of AI risk?",
        "harm_distribution_basis": "What kind of harm is likely to occured from the AI system usage?",
        "sector_of_deployment": "Which sector(s) is the AI system deployed in?",
    }
)


def load_label_mapping_from_csv(csv_path, label_column):
    """Load a feature-to-label mapping from a CSV with a header row."""
    if not os.path.exists(csv_path):
        return {}

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error = None

    for encoding in encodings_to_try:
        try:
            labels_df = pd.read_csv(
                csv_path,
                header=0,
                dtype=str,
                encoding=encoding,
            )
            if "features" not in labels_df.columns or label_column not in labels_df.columns:
                raise ValueError(f"CSV must contain 'features' and '{label_column}' columns")

            labels_df = labels_df[["features", label_column]].dropna(subset=["features", label_column])
            labels_df["features"] = labels_df["features"].str.strip()
            labels_df[label_column] = labels_df[label_column].str.strip()
            labels_df = labels_df[(labels_df["features"] != "") & (labels_df[label_column] != "")]
            return dict(zip(labels_df["features"], labels_df[label_column]))
        except UnicodeDecodeError as error:
            last_error = error
            continue
        except Exception as error:
            st.warning(f"Could not load feature labels config: {error}")
            return {}

    st.warning(f"Could not load feature labels config: {last_error}")
    return {}


feature_labels_config_path = os.path.join(
    os.path.dirname(__file__), "..", "model", "features_label_config.csv"
)
FEATURE_DISPLAY_LABELS.update(load_label_mapping_from_csv(feature_labels_config_path, "form_label"))
RISK_EXPOSURE_DISPLAY_LABELS = load_label_mapping_from_csv(feature_labels_config_path, "risk_exposure_label")


def get_risk_exposure_label(feature_name):
    return RISK_EXPOSURE_DISPLAY_LABELS.get(feature_name, get_feature_label(feature_name))

# List the features that should use the shared continuous 0.5-step input.
CONTINUOUS_FEATURES = [
    "physical_objects",
    "impact_on_critical_services",
    "involving_minor",
    "detrimental_content",
    "protected_characteristic",
    "public_sector_deployment",
    "autonomy_level_encoded",
    "rights_violation"
]

for feature_name in CONTINUOUS_FEATURES:
    FEATURE_INPUT_FORMATS[feature_name] = {
        "type": "continuous",
        "step": 0.5,
        "min": 0.0,
        "max": 1.0
    }

def is_checkbox_feature(feature_name):
    return any(
        feature_name == feature_key or feature_name.startswith(f"{feature_key}")
        for feature_key in CHECKBOX_GROUP_KEYS
    )


def get_checkbox_group_name(feature_name):
    for feature_key in CHECKBOX_GROUP_KEYS:
        if feature_name == feature_key or feature_name.startswith(f"{feature_key}"):
            return feature_key
    return feature_name


def get_feature_label(feature_name):
    return FEATURE_DISPLAY_LABELS.get(feature_name, default_feature_label(feature_name))


for feature_name in feature_names:
    if is_checkbox_feature(feature_name):
        FEATURE_INPUT_FORMATS[feature_name] = {"type": "checkbox"}


def render_feature_input(feature_name, index):
    feature_format = FEATURE_INPUT_FORMATS.get(feature_name, {"type": "continuous", "step": 0.5})
    label = get_feature_label(feature_name)

    if feature_format.get("type") == "binary":
        return st.radio(
            label=label,
            options=[0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
            horizontal=True,
            key=f"feature_{index}",
        )

    if feature_format.get("type") == "checkbox":
        return int(
            st.checkbox(
                label=label,
                key=f"feature_{index}",
            )
        )

    return st.radio(
        label=label,
        options=[0.0, 0.5, 1.0],
        format_func=lambda value: {
            0.0: "No",
            0.5: "Partially",
            1.0: "Yes",
        }[float(value)],
        horizontal=True,
        key=f"feature_{index}",
    )
# Create the input template UI
st.title("AI Risk Assessment - Data Input")

with st.form("input_form"):
    st.subheader("Enter Feature Values")
    
    # Create input fields dynamically based on model features
    input_values = []
    cols = st.columns(1)
    active_checkbox_group = None
    
    for i, feature_name in enumerate(feature_names):
        with cols[i % 1]: # Adjust the number of columns as needed
            feature_format = FEATURE_INPUT_FORMATS.get(feature_name, {"type": "continuous", "step": 0.5})

            if feature_format.get("type") == "checkbox":
                group_name = get_checkbox_group_name(feature_name)
                group_label = FEATURE_DISPLAY_LABELS.get(group_name, default_feature_label(group_name))

                if group_name != active_checkbox_group:
                    st.write(group_label)
                    active_checkbox_group = group_name

                value = int(
                    st.checkbox(
                        label=get_feature_label(feature_name),
                        key=f"feature_{i}",
                    )
                )
            else:
                active_checkbox_group = None
                value = render_feature_input(feature_name, i)
            input_values.append(value)
    
    # Submit button
    submitted = st.form_submit_button("Start Assess AI Risks")
    
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
            st.success("Assessment Complete!")
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
                positive_input_features["feature"] = positive_input_features["feature"].map(get_risk_exposure_label)
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
                    positive_features["feature"] = positive_features["feature"].map(get_risk_exposure_label)
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

