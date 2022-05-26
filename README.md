# Camera Calibration
Implement of camera calibration with opencv

## Usage

```python
from calibrate import CameraCalibrator
from preprocessing import grayscale
import cv2

cc = CameraCalibrator(grayscale)
image_path = "images/Image_20220525114314421.bmp"; nx, ny = 7, 6
img = cv2.imread(image_path)
undist, points = cc.calibrate(img, nx, ny)
length = getlength(points)
viz = viz_poly(undist, points)
print(f"Top / Bottom: {length['top']/length['bottom']:.7f}")
print(f"Left / Right: {length['left']/length['right']:.7f}")
cv2.imwrite("calibrate.png", viz)
```

