from darkflow.net.build import TFNet
import cv2
from time import time

options = {"model": "yolo-obj.cfg", "load": "yolo-obj_2000.weights", "threshold": 0.1, "gpu": 1.0}

tfnet = TFNet(options)

for i in range (5):
  imgcv = cv2.imread("./left0315.jpg")
  start = time()
  result = tfnet.return_predict(imgcv)
  print(time() - start)
  print(result)