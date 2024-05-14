from deepface import DeepFace
import cv2
# import matplotlib.pyplot as plt

img1 = cv2.imread('public/DJI_0617.JPG')

img2 = cv2.imread('public/DJI_0657.JPG')


result = DeepFace.verify(img1,img2)

print("is same face:", result['verified'])

