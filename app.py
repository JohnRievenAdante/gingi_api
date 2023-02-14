import uvicorn
import pickle
from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel,parse_obj_as
class data2(BaseModel):
    image: list

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'hello'}


@app.post('/prediction')
def get_image_category(data:data2):
    received = data.image
    #img4=received['image']
    
    # get HOG features from greyscale image
    # combine color and hog features into a single array
    
    ss = StandardScaler()
    bees_stand = ss.fit_transform(received)
    gum_pca = ss.fit_transform(bees_stand)
    loaded_model = pickle.load(open("finalized_model.pkl", 'rb'))
    predict=loaded_model.predict(gum_pca)
    return {'transformed': str(predict[0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
