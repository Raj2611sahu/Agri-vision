import base64
import requests
from io import BytesIO
from PIL import Image




def process_plant_image(image_file):
    img = Image.open(image_file).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "model": "gemma4:e4b",
        "prompt": (
            "You are an expert botanist.\n"
            "Analyze this plant image and identify visible symptoms only.\n"
            "List possible diseases, nutrient deficiencies, or pest damage.\n"
            "Do NOT suggest treatment yet."
        ),
        "images": [img_str],
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]

def refine_with_knowledge(context, vision_output):

    prompt = f"""
You are an agricultural scientist.

IMAGE-BASED OBSERVATION:
{vision_output}

REFERENCE KNOWLEDGE:
{context}

TASK:
1. Confirm or reject the image-based diagnosis using the reference knowledge.
2. Provide the most likely issue.
3. Suggest organic treatments only.
4. Mention uncertainty if evidence is weak.

Answer clearly and concisely.
"""

    payload = {
        "model": "gemma4:e4b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]