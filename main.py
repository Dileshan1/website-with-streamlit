import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

st.header("Diabetes Detection App Using Machine Learning")

image=Image.open("diab.jpg")
st.image(image)

data = pd.read_csv("diabetes.csv")
st.subheader("Data")
st.dataframe(data)


# Data Summaries
st.subheader("Summary of Numerical Data")
st.write(data.iloc[:,:8].describe())

st.subheader("Summary of Numerical Data for Diabetes & Non-Diabetes")
st.write(data.groupby("Outcome").agg(["mean","median","max","min"]))


# Data Visualization
st.subheader("Distribution of Glucose")

fig, ax = plt.subplots()
ax.hist(data["Glucose"], bins=20)
st.pyplot(fig)                  # can't use "plt.hist(data["Glucose"], bins=20)"