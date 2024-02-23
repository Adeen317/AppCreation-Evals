import streamlit as st
import os
from dotenv import load_dotenv
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# Get Clarifai API key from environment variables
clarifai_pat = os.getenv('CLARIFAI_PAT')

def generate_dream_image(prompt):
    inference_params = dict(quality="standard", size='1024x1024')
    model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
    output_base64 = model_prediction.outputs[0].data.image.base64
    return output_base64

def analyze_dream(output_base64, description):
    prompt = "What type of dream it is like in a good or bad way and what is the dream trying to portray? Do not use the word image or picture in the description rather explain it as a dream"
    inference_params = dict(temperature=0.4, max_tokens=100)
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision").predict(inputs=[Inputs.get_multimodal_input(input_id="", image_bytes=output_base64, raw_text=prompt)], inference_params=inference_params)
    return model_prediction.outputs[0].data.text.raw

def main():
    st.title('Dream Vue')
    query = st.text_input('Enter your dream description:')
    if st.button('Analyze'):
        output_base64 = generate_dream_image(query)
        image = Image.open(io.BytesIO(output_base64))
        st.image(image, caption='Generated Dream Image', use_column_width=True)
        result = analyze_dream(output_base64, query)
        st.write('Dream analysis:')
        st.write(result)

if __name__ == '__main__':
    main()
