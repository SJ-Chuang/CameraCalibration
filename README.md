# Camera Calibration
Implement of camera calibration with opencv

## Usage

**Arguments**

- image_path: path to an image.
- nx: number of the inside corners along the x-axis.
- ny: number of the inside corners along the y-axis.

```python
from calibrate import CameraCalibrator, getlength, viz_poly
from preprocessing import grayscale
import cv2

image_path = "images/Image_20220525114314421.bmp"; nx, ny = 7, 6

cc = CameraCalibrator(grayscale)
img = cv2.imread(image_path)
undist, points = cc.calibrate(img, nx, ny)
length = getlength(points)
viz = viz_poly(undist, points)
print(f"Top / Bottom: {length['top']/length['bottom']:.7f}")
print(f"Left / Right: {length['left']/length['right']:.7f}")
cv2.imwrite("calibrate.png", viz)
```

