from calibrate import CameraCalibrator, getlength, viz_poly
from preprocessing import grayscale
import cv2

cc = CameraCalibrator(grayscale)
image_path = "images/Image_20220525114314421.bmp"; nx, ny = 7, 6
img = cv2.imread(image_path)
cc.calibrate([img], nx, ny)
cc.save("config.ini")

cc.load("config.ini")
undist = cc.undistort(img)
points = cc.findcorners(img, nx, ny)
length = getlength(points)
viz = viz_poly(undist, points)
print(f"Top / Bottom: {length['top']/length['bottom']:.7f}")
print(f"Left / Right: {length['left']/length['right']:.7f}")
cv2.imwrite("calibrate.png", viz)