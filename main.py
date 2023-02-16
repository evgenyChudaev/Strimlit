# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:43:29 2023

@author: eugen
"""

import streamlit as st
from joblib import load
from sklearn import tree

st.title('Test')
LABELS = ['setosa', 'versicolor', 'virginica']

clf = load("DT.joblib")

sp_l = st.slider('sepal len (cm)', min_value = 0, max_value = 10)
sp_w = st.slider('sepal width (cm)', min_value = 0, max_value = 10)
pe_l = st.slider('pital len (cm)', min_value = 0, max_value = 10)
pe_w = st.slider('pital width (cm)', min_value = 0, max_value = 10)

prediction = clf.predict([[sp_l, sp_w, pe_l, pe_w]])

st.write(LABELS[prediction[0]])
