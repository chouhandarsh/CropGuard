from fastapi import FastAPI, File, UploadFile
import numpy as np
import BytesIO 
from PIL import  Image
import uvicorn
app = FastAPI()
@app.get("/ping")
async def ping():
    return "Hello server is Live"
@app.post("/predict")
def read_file_as_image(data)->np.array:
    image =np.array(Image.open(BytesIO(data)))
async def predict(
    file: UploadFile=(...)
):
    bytes= await file.read()
    return

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)