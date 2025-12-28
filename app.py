import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

final_X = X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)

st.sidebar.title('Select House features: ')
st.sidebar.image('https://cdn.dribbble.com/userupload/20000742/file/original-aaf23458355a156d0cf85b8217a5065a.gif')
all_value = []
for i in final_x:
   result = st.sidebar.slider(f'Select {1} value')
  all_value.append(result)

st.write(all_value)

