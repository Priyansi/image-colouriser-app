import json
import requests
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


def predict_array(PATH):
    resp = requests.post("https://image-colouriser.herokuapp.com/predict",
                         files={'file': open(PATH, 'rb')})

    return resp.text


img_dict = json.loads(predict_array('test.jpg'))
img_list = img_dict['prediction']
img_arr = np.array(img_list)
plt.imsave('prediction.png', img_arr)
