"""import uvicorn
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
    return {'message': "hello"}


@app.post('/prediction')
def get_image_category(data:data2):
    received = data.image
    #img4=received['image']
    received=np.array(received)
    print(received.shape)
    
    # get HOG features from greyscale image
    # combine color and hog features into a single array
    color_features = received.flatten()
    # get HOG features from greyscale image
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    ok=flat_features.reshape(1, -1)
    ss = StandardScaler()
    bees_stand = ss.fit_transform(ok)
    gum_pca = ss.fit_transform(bees_stand)
    #loaded_model = pickle.load(open("finalized_model.pkl", 'rb'))
    #predict=loaded_model.predict(gum_pca)
    return {'transformed': gum_pca.tolist()}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000)

"""
from flask import Flask, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app=Flask(__name__)

@app.route('/prediction',methods=['POST'])
def upload_image():
    image_bytes=request.data
    image=np.frombuffer(image_bytes,dtype=np.uint8).reshape((256,256))
    color_features = image.flatten()
    # get HOG features from greyscale image
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    ok=flat_features.reshape(1, -1)
    ss = StandardScaler()
    bees_stand = ss.fit_transform(ok)
    gum_pca = ss.fit_transform(bees_stand)
    loaded_model = pickle.load(open("finalized_model.pkl", 'rb'))
    predict=loaded_model.predict(gum_pca)
    return {'transformed': str(predict[0])}
    #return {'transformed': image.tolist()}


if __name__ == '__main__':
    app.run()
