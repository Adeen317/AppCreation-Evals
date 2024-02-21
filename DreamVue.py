from dotenv import load_dotenv
load_dotenv()
import os
clarifai_pat=os.getenv('CLARIFAI_PAT')
from clarifai.client.model import Model
from clarifai.client.input import Inputs

prompt = input("Visualize your dream here: ")


inference_params = dict(quality="standard", size= '1024x1024')

# Model Predict
model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)

output_base64 = model_prediction.outputs[0].data.image.base64

with open('image.png', 'wb') as f:
    f.write(output_base64)



prompt = "What type of dream it is like in a good or bad way and what is the dream trying to potray? Do not use the word image or picture in the description rather explain it as a dream"
inference_params = dict(temperature=0.4, max_tokens=100)

model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision").predict(inputs = [Inputs.get_multimodal_input(input_id="", image_bytes = output_base64, raw_text=prompt)], inference_params=inference_params)
print(model_prediction.outputs[0].data.text.raw)






