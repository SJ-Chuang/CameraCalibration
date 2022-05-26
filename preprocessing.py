import cv2

"""
Preprocessing functions
"""
def erosion(img, kernel=(21,21), iterations=11):
    binary = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 255)
    return cv2.erode(binary, kernel, iterations=iterations)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)