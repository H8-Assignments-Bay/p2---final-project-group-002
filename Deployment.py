import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
st.set_page_config(layout="centered", page_title="Fashion CNN")



# Load model
model = tf.keras.models.load_model("my_model_imp.h5")
img = None
LABEL = 'Apparel Set', 'Bottomwear', 'Dress', 'Flipflop', 'Innerwear', 'Sandal', 'Shoes', 'Socks', 'Topwear'


img_height= 250 
img_width= 250 

# Function Prediction
def load_image(img):
    img = tf.keras.utils.load_img(img, target_size=(img_height,img_width),grayscale=True) 
    x = tf.keras.utils.img_to_array(img) 
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    res = model.predict(images)
    res = tf.keras.layers.Softmax()(res)

    return np.argmax(res, axis = 1)[0],res

# Homepage
gambar = Image.open('ds_2.png')
st.image(gambar, use_column_width = True, caption='GoodStyle Fashion Recommendation')

#st.title("GOODSTYLE")
st.markdown("<h1 style='text-align: center; color: #964C5A;'>GOODSTYLE</h1>", unsafe_allow_html=True)

st.subheader("Profile")
st.write("""Goodstyle was born with a new idea that was inspired by an interaction between brands, 
             communities and collaborations involving people in the fashion world. A journey from year to year which finally becomes a new perspective for us to celebrate, 
             starting from a history, a phenomenon of fashion culture and the achievements that the Goodstyle has made so far.""")

# Upload image section
st.subheader('''Let's find out which apparel the photo is''')
prediction = st.file_uploader("Upload an Apparel Photo here", type=["jpg", "png", "jpeg"])
if prediction is not None: 
    st.image(prediction, use_column_width='auto')
    btn = st.button('Predict')
    if btn:
        pred = load_image(prediction)
        if pred :
            st.success('The image is {}'.format(LABEL[pred[0]]))
        else:
            st.error('Error')