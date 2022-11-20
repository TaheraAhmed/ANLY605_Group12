# pip install -r /path/to/requirements.txt

import streamlit as st
import pandas as pd
import joblib

import altair as alt

import plotly.express as px
import plotly.graph_objects as go

# Title
st.header("InstaCart Reordering Prediction App")

df = pd.read_csv('train_dataset.csv')

fig = alt.Chart(df).mark_boxplot().encode(
    x='reordered:N', y='uxp_reorder_ratio:Q', color='reordered:N')
        
st.altair_chart(fig, use_container_width=True)


# Input bar 1
u_p_total = st.number_input("Enter the number of times a user has bought the product")

# Input bar 2
uxp_reorder_ratio = st.number_input("Enter the estimated probability of how frequently a user bought the product")

# Input bar 3
u_total_orders= st.number_input("Enter the number of orders placed by a user")

# Input bar 4
u_reordered_ratio = st.number_input("Enter how frequently has a user reordered products")

# Input bar 5
p_total=st.number_input("Enter the number of times a product has been purchased")

# Input bar 6
p_reordered_ratio=st.number_input("Enter the estimated probability of porduct being reordered")

# What does the predicted value mean?
st.text('1 means Reordered, 0 means Not reordered')

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("StackedPickle.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[u_p_total, uxp_reorder_ratio, u_total_orders,u_reordered_ratio,p_total,p_reordered_ratio]], 
                     columns = ["u_p_total", "uxp_reorder_ratio", "u_total_orders",'u_reordered_ratio',"p_total","p_reordered_ratio"])


    
    # Get prediction
    prediction = clf.predict(X)[0]
    X['reordered']=prediction
    
    # Output prediction
    st.text(f"This product will be {prediction}")
