import numpy as np
from mss import mss
import cv2
import matplotlib.pyplot as plt

def ShowImage(image):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()
def ProcessImage(image, save = False, filename = "saved.png"):
    height, width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    xdiff = int((width - 960) / 2)
    ydiff = int((height - 960) / 2)
    image = image[ydiff:(height - ydiff), xdiff:(width - xdiff)]
    image = cv2.resize(image, (32,32))
    image = np.ravel(image)
    image = image / 255.0
    return image

def GetImage(saveprocessed = False, filename = "GeoDash.png"):
    filename0 = "Images/" + filename
    with mss() as sct:
        sct.shot(output=filename0)
    image = cv2.imread(filename0)
    if saveprocessed:
        p_image = ProcessImage(image, True, filename)
    else:
        p_image = ProcessImage(image)
    return p_image