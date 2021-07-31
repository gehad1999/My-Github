import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
import base64


im = Image.open(BytesIO(base64.b64decode(open('n.txt', 'rb').read())))
plt.imshow(im)
plt.show()