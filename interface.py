import streamlit as st
import pandas as pd
import pickle
import numpy as np

# sidebar with parameters

st.sidebar.title("Parameters")

O = st.sidebar.number_input('Net Fraction Revolving Burden', 
                      min_value=-10, max_value=1000)

E = st.sidebar.number_input('External Risk Estimate', 
                      min_value=-10, max_value=1000)

AM = st.sidebar.number_input('Average Months in File', 
                      min_value=-20, max_value=1000)

ND = st.sidebar.number_input('Max Delinquency/Public Records Last 12 Months', 
                      min_value=-9, max_value=1000)

RD = st.sidebar.number_input('Number Bank/Natl Trades with high utilization ratio', 
                      min_value=-10, max_value=1000)

submit = st.sidebar.button("Submit")

# header
st.header("HELOC Risk Performance Prediction")

#result
model = pickle.load(open('boosting_5.p','rb'))

prediction = model.predict([[E,O,AM,ND,RD]])[0]

if prediction == 1:
    result = "good"
else:
    result = "bad"

st.text('The risk performance is %s'%(result))

#charts
input_data = np.array([[E, O, AM, ND, RD]])
probas = model.predict_proba(input_data)[0]
categories = ["Bad" if label == 0 else "Good" for label in model.classes_]
probas_dict = dict(zip(categories, probas))
df = pd.DataFrame.from_dict(probas_dict, orient='index', columns=['Probability'])
st.subheader("Predicted Probabilities")
st.write(df)














