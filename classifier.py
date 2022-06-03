from importlib_metadata import version
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps 
import cv2
x = np.load('image.npz')["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
x_train , x_test , y_train , y_test = train_test_split(x , y , random_state=9 , train_size=7500 , test_size=2500)
x_train_scaled = x_train/255
x_test_scaled  = x_test/255
model = LogisticRegression(solver="saga" , multi_class="multinomial")
model.fit(x_train_scaled , y_train)

def getPred(image):
        im_pil = Image.open(image)
        im_bw = im_pil.convert("L")
        im_bw_resized = im_bw.resize((28,28) , Image.ANTIALIAS)
        pixel_filter = 20
        min_pixel = np.percentile(im_bw_resized , pixel_filter)
        im_bw_resized_inverted = np.clip(im_bw_resized - min_pixel,0,255)
        max_pixel = np.max(im_bw_resized)
        im_bw_resized_inverted_scaled = np.asarray(im_bw_resized_inverted)/max_pixel
        test = np.array(im_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = model.predict(test)
        return test_pred[0]