import cv2
import matplotlib.pyplot as plt

dog_img = cv2.imread('_OPEN_CV - COMPUTER VISION/DATA/sammy_face.jpg')
dog_img = cv2.cvtColor(dog_img,cv2.COLOR_BGR2RGB)

blured_dog_img_median = cv2.medianBlur(dog_img,5)
default_blur = cv2.blur(dog_img,(5,5))
gaussian_blur = cv2.GaussianBlur(dog_img,(5,5),10)

edged_img = cv2.Canny(gaussian_blur,80,120)
plt.imshow(edged_img)