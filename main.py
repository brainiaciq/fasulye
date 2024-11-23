from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
from pathlib import Path
import cv2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost"
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = keras.models.load_model("C:/Fasulye/saved_models/2")
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    
keras.backend.set_image_data_format()

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    image = read_file_as_image(await file.read())
    image = cv2.resize(image, (256,256))
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host= "localhost", port=8000)