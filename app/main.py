from fastapi import FastAPI, File, UploadFile
import json
import numpy as np
import requests
from io import BytesIO 
from PIL import  Image
import tensorflow as tf
TF_SERVING_URL = "http://localhost:8501/v1/models/potato_disease:predict"
"""MODEL = tf.keras.models.load_model(
    r"D:\Projects\Potato_Disease\models\potato_model.keras"
)"""
CLASS_NAME = ['Early Blight','Late Blight','Healthy']
def read_file_as_image(data)->np.array:
    image =np.array(Image.open(BytesIO(data)))
    return image
import uvicorn
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/ping")
async def ping():
    return "Hello server is Live"
@app.post("/predict")
async def predict(
    file: UploadFile=File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,axis=0)
    # prediction=MODEL.predict(img_batch)
    json_data={
        'instances':img_batch.tolist()
    }
    response = requests.post(TF_SERVING_URL,json=json_data)
    prediction = np.array(response.json()["predictions"])
    confidence = float(np.max(prediction))
    
    return {
        "class": CLASS_NAME[np.argmax(prediction[0])],
        "confidence": confidence
    }


if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8080)