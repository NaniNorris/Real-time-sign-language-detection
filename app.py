import streamlit as st
import numpy as np
import pandas as pd
from landmarks import holostic_model
from preprocess import load_data,get_interesting_idx
import matplotlib.pyplot as plt
from tensorflow.keras import models
import tensorflow as tf
import pickle

st.title("Sign language detection model")

with open("Label_encoder_5.pkl",'rb') as file:
    label_encode = pickle.load(file)

@st.cache_resource
def load_model():
    model = models.load_model("5_action_v2_acc_94_valacc_81")
    idx = get_interesting_idx()
    data_load = load_data(idx)
    mp_model = holostic_model()
    return model,idx,data_load,mp_model

model,idx,data_load,mp_model = load_model()

value = mp_model.extract_values("Dad.mp4",display=True)
value = data_load.load_no_sign_data(value)
pred = model.predict(tf.expand_dims(value,0))
pred = label_encode.inverse_transform([[np.argmax(pred.squeeze())]])
st.write(f"Acttual sign : 'Dad.mp4' Predicted sign : {pred[0][0]}")