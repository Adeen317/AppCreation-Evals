from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from clarifai.client.model import Model
from clarifai.client.input import Inputs

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get Clarifai API key from environment variables
clarifai_pat = os.getenv('CLARIFAI_PAT')

def generate_dream_image(prompt):
    inference_params = dict(quality="standard", size='1024x1024')
    model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
    output_base64 = model_prediction.outputs[0].data.image.base64
    with open('image.png', 'wb') as f:
        f.write(output_base64)
    return output_base64

def analyze_dream(output_base64, description):
    prompt = "What type of dream it is like in a good or bad way and what is the dream trying to potray? Do not use the word image or picture in the description rather explain it as a dream"
    inference_params = dict(temperature=0.4, max_tokens=100)
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision").predict(inputs=[Inputs.get_multimodal_input(input_id="", image_bytes=output_base64, raw_text=prompt)], inference_params=inference_params)
    return model_prediction.outputs[0].data.text.raw

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['q']
    output_base64 = generate_dream_image(query)
    result = analyze_dream(output_base64, query)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
