import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

# load trained model , scaler , ohe , lable
model = tf.keras.models.load_model('model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
ohe = pickle.load(open('ohe_encoder_geo.pkl', 'rb'))
lable = pickle.load(open('lable_encoder_gender.pkl', 'rb'))


## streamlit app

st.title("Customer Churn Prediction")

## all inputs

gender = st.selectbox("Gender",lable.classes_)
geography = st.selectbox("Geography",ohe.categories_[0])
has_cr = st.selectbox("Has Credit Card",["Yes","No"])
is_active = st.selectbox("Is Active Member",["Yes","No"])
age = st.slider("Age",18,90)
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
credit_score = st.number_input("Credit Score")
balance = st.number_input("Balance")
salary = st.number_input("Estimated Salary")

## prepare input data

input_data = {
    'CreditScore' : [credit_score],
    'Gender' : [lable.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr],
    'IsActiveMember' : [is_active],
    'EstimatedSalary' : [salary],
}

input_data_df = pd.DataFrame(input_data)


## encode Geography

geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out())
input_data_df = pd.concat([input_data_df.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_df['HasCrCard'] = input_data_df['HasCrCard'].map({'Yes': 1, 'No': 0})
input_data_df['IsActiveMember'] = input_data_df['IsActiveMember'].map({'Yes': 1, 'No': 0})

st.write(input_data_df)
## scaling data

scaled_data = scaler.transform(input_data_df)

# ## predcition

if st.button("Predict"):
    prediction = model.predict(scaled_data)
    prediction_proba =prediction[0][0]
    if prediction_proba > 0.6:
        st.error(f"Probability Of Leaving: {prediction_proba:.2f}")
        st.write("Customer Will Leave")
    else:
        st.success(f"Probability Of Leaving: {prediction_proba:.2f}")
        st.write("Customer Will Not Leave")