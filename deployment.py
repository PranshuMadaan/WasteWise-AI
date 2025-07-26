import streamlit as st
import pickle
import Orange

# Load model
with open("C:/Users/user/Desktop/WasteWise AI/WasteWise AI.pkcls", "rb") as file:
    model = pickle.load(file)

domain = model.domain

st.title("WasteWise AI")
st.write("By TEAM APEX")

# Inputs
food_prepared = st.number_input("Food Prepared", value=0.0)
food_sold = st.number_input("Food Sold", value=0.0)

day_of_week = st.selectbox(
    "Day of the Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

month = st.selectbox(
    "Month",
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"]
)

# Show domain info for debugging
if st.button("Show Model Domain"):
    for i, var in enumerate(domain):
        st.write(f"Feature {i}: name={var.name}, type={'Discrete' if var.is_discrete else 'Continuous'}")
        if var.is_discrete:
            st.write(f"  Possible values: {list(var.values)}")

if st.button("Predict"):
    inputs = [food_sold, food_prepared, day_of_week, month]
    row = []
    try:
        for var, val in zip(domain, inputs):
            if var.is_discrete:
                # map string (e.g., "Monday") to index
                idx = list(var.values).index(val)
                row.append(idx)
            else:
                # already numeric
                row.append(float(val))

        table = Orange.data.Table(domain, [row])

        y_pred = model(table)
        result = y_pred[0].value if hasattr(y_pred[0], 'value') else y_pred[0]

        st.success(f"Predicted Waste: {result}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
