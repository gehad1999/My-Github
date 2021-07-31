
import numpy as np

import matplotlib.pyplot as plt
import cv2
import numpy as np

import matplotlib.pyplot as plt
import requests
from io import BytesIO

from scipy import misc
from PIL import Image, ImageFile



img=cv2.imread('Remind.jpg')
m=1
vv=None
    
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, encimg = cv2.imencode('Remind.jpg', img, encode_param)
num_initial=len(encimg)
padding = { 0:0, 1:2, 2:1 }[num_initial % 3]
for i in range((int(len(encimg)/40))+1):
    vv=np.append(encimg[i*10:(i+1)*10],vv)
    print( encimg[i*10:(i+1)*10])
    nn=cv2.imdecode(encimg,1)


    plt.imshow(vv)
    plt.show()