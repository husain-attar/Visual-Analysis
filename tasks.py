from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import streamlit as st
import requests
from PIL import Image

# Load processor and model
processor_vqa = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
model_vqa = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")
processor_coco = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model_coco = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# def process_task(task, image):
#     if task == "Visual Question Answering":
#         question = st.text_input("Enter your question about the image:")
#         if question:
#             pixel_values = processor_vqa(images=image, return_tensors="pt").pixel_values
#             input_ids = processor_vqa(text=question, add_special_tokens=False).input_ids
#             input_ids = [processor_vqa.tokenizer.cls_token_id] + input_ids
#             input_ids = torch.tensor(input_ids).unsqueeze(0)
#             generated_ids = model_vqa.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=100)
#             answer = processor_vqa.batch_decode(generated_ids, skip_special_tokens=True)[0]
#             st.subheader("Answer:")
#             st.write(answer)
def process_task(task, image):
    if task == "Visual Question Answering":
        question = st.text_input("Enter your question about the image:")
        if question:
            pixel_values = processor_vqa(images=image, return_tensors="pt").pixel_values
            input_ids = processor_vqa(text=question, add_special_tokens=False).input_ids
            input_ids = [processor_vqa.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            generated_ids = model_vqa.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=100)
            generated_text = processor_vqa.batch_decode(generated_ids, skip_special_tokens=True)[0]

            answer = generated_text.split(question)[-1].strip()
            
            st.subheader("Answer:")
            st.write(answer)


    elif task == "Image Captioning":
        pixel_values = processor_coco(images=image, return_tensors="pt").pixel_values
        generated_ids = model_coco.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor_coco.batch_decode(generated_ids, skip_special_tokens=True)[0]
        st.subheader("Caption:")
        st.write(generated_caption)
