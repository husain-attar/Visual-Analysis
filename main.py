import streamlit as st
import requests
from PIL import Image
from tasks import process_task

st.title("Image and Text Analysis")

# Sidebar for selecting the task
task = st.sidebar.selectbox("Select Task", ["Visual Question Answering", "Image Captioning"])

# Main content
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    process_task(task, image)
else:
    st.warning("Please upload an image.")

# Or allow the user to input an image URL
image_url = st.text_input("Enter Image URL:")
if image_url:
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        process_task(task, image)
    except Exception as e:
        st.error("Error loading image from URL. Please check the URL.")
