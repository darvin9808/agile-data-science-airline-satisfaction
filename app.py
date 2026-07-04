
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Airline Passenger Satisfaction Dashboard", layout="wide")

st.title("✈️ Airline Passenger Satisfaction Dashboard")

st.write("""
This dashboard supports airline stakeholders in analysing passenger satisfaction patterns
and predicting passenger satisfaction using the improved XGBoost model from Sprint 3.
""")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_excel("train dataset.xlsx")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
    df["Arrival Delay in Minutes"].median()
)

# ============================================================
# Sidebar Interactive Filters
# ============================================================

st.sidebar.header("Interactive Filters")

selected_class = st.sidebar.selectbox(
    "Select Travel Class",
    ["All"] + list(df["Class"].unique())
)

selected_age = st.sidebar.slider(
    "Select Passenger Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (20, 60)
)

filtered_df = df.copy()

if selected_class != "All":
    filtered_df = filtered_df[filtered_df["Class"] == selected_class]

filtered_df = filtered_df[
    (filtered_df["Age"] >= selected_age[0]) &
    (filtered_df["Age"] <= selected_age[1])
]

st.subheader("Filtered Passenger Dataset")
st.dataframe(filtered_df.head())

# ============================================================
# Visualization 1
# ============================================================

st.subheader("Visualization 1: Passenger Satisfaction Distribution")

fig1, ax1 = plt.subplots(figsize=(6, 4))
filtered_df["satisfaction"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Satisfaction")
ax1.set_ylabel("Number of Passengers")
ax1.set_title("Passenger Satisfaction Distribution")
st.pyplot(fig1)

# ============================================================
# Visualization 2
# ============================================================

st.subheader("Visualization 2: Passenger Satisfaction by Travel Class")

fig2, ax2 = plt.subplots(figsize=(6, 4))
pd.crosstab(filtered_df["Class"], filtered_df["satisfaction"]).plot(kind="bar", ax=ax2)
ax2.set_xlabel("Travel Class")
ax2.set_ylabel("Number of Passengers")
ax2.set_title("Passenger Satisfaction by Travel Class")
st.pyplot(fig2)

# ============================================================
# Visualization 3
# ============================================================

st.subheader("Visualization 3: Average Arrival Delay by Satisfaction")

fig3, ax3 = plt.subplots(figsize=(6, 4))
filtered_df.groupby("satisfaction")["Arrival Delay in Minutes"].mean().plot(kind="bar", ax=ax3)
ax3.set_xlabel("Satisfaction")
ax3.set_ylabel("Average Delay in Minutes")
ax3.set_title("Average Arrival Delay by Satisfaction")
st.pyplot(fig3)

# ============================================================
# Analytical Summary
# ============================================================

st.subheader("Analytical Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric("Average Flight Distance", round(filtered_df["Flight Distance"].mean(), 2))

with col2:
    st.metric("Average Passenger Age", round(filtered_df["Age"].mean(), 1))

# ============================================================
# Model Training Function
# ============================================================

@st.cache_resource
def train_model(data):

    df_fe = data.copy()

    label_encoders = {}

    categorical_columns = df_fe.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        le = LabelEncoder()
        df_fe[column] = le.fit_transform(df_fe[column])
        label_encoders[column] = le

    numerical_columns = [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes"
    ]

    scaler = StandardScaler()
    df_fe[numerical_columns] = scaler.fit_transform(df_fe[numerical_columns])

    X = df_fe.drop("satisfaction", axis=1)
    y = df_fe["satisfaction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.10,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model, label_encoders, scaler, X.columns, numerical_columns, data


model, label_encoders, scaler, feature_columns, numerical_columns, original_df = train_model(df)

# ============================================================
# Predictive Output
# ============================================================

st.subheader("Predictive Output: Passenger Satisfaction Prediction")

st.write("Enter passenger details below to predict whether the passenger is likely to be satisfied.")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", df["Gender"].unique())
    customer_type = st.selectbox("Customer Type", df["Customer Type"].unique())
    age_input = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 35)

with col2:
    travel_type = st.selectbox("Type of Travel", df["Type of Travel"].unique())
    travel_class = st.selectbox("Class", df["Class"].unique())
    flight_distance = st.number_input("Flight Distance", min_value=0, value=1000)

with col3:
    online_boarding = st.slider("Online Boarding Rating", 0, 5, 3)
    seat_comfort = st.slider("Seat Comfort Rating", 0, 5, 3)
    inflight_entertainment = st.slider("Inflight Entertainment Rating", 0, 5, 3)

if st.button("Predict Satisfaction"):

    input_data = {}

    for col in feature_columns:
        if df[col].dtype == "object":
            input_data[col] = df[col].mode()[0]
        else:
            input_data[col] = df[col].median()

    input_data["Gender"] = gender
    input_data["Customer Type"] = customer_type
    input_data["Age"] = age_input
    input_data["Type of Travel"] = travel_type
    input_data["Class"] = travel_class
    input_data["Flight Distance"] = flight_distance
    input_data["Online boarding"] = online_boarding
    input_data["Seat comfort"] = seat_comfort
    input_data["Inflight entertainment"] = inflight_entertainment

    input_df = pd.DataFrame([input_data])

    for column, le in label_encoders.items():
        if column != "satisfaction" and column in input_df.columns:
            input_df[column] = le.transform(input_df[column])

    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]

    satisfaction_label = label_encoders["satisfaction"].inverse_transform([prediction])[0]

    st.success(f"Predicted Passenger Satisfaction: {satisfaction_label}")
