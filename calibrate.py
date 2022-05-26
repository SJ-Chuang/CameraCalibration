import os, cv2
import numpy as np

def viz_poly(img, points, line_width: int=5):
    viz = img.copy()
    cv2.polylines(viz, pts=[points.astype(int)], isClosed=True, color=(0, 0, 255), thickness=line_width)
    for i in range(len(points)):
        L = np.linalg.norm(points[i]-points[i-1]).astype(int)
        w, h = cv2.getTextSize(f"{L}pix", 16, 1, 2)[0]
        x, y = np.mean([points[i], points[i-1]], 0).astype(int)
        x -= w // 2
        y -= h // 2
        cv2.putText(viz, f"{L}pix", (x, y), 16, 3, (0, 0, 0), 7)
        cv2.putText(viz, f"{L}pix", (x, y), 16, 3, (255, 255, 255), 3)
    return viz

def getlength(points):
    length = []
    for i in range(len(points)):
        length.append(np.linalg.norm(points[i]-points[i-1]).astype(int))
    
    return {"top": length[2], "bottom": length[0], "left": length[1], "right": length[3]}

"""
CameraCalibrator Object
Args:
    preprocessing (callable): a callable function to preprocessing the input image.
"""
class CameraCalibrator:
    def __init__(self, preprocessing: callable=grayscale):
        self.preprocessing = preprocessing
    
    def calibrate(self, img, nx, ny):
        """
        Calibrate camera
        Args:
            img (np.ndarray): input image.
            nx (int): number of inside corners in x.
            ny (int): number of inside corners in y.
        
        Returns:
            visualization and length of sides.
        """
        pre = self.preprocessing(img)
        ret, corners = cv2.findChessboardCorners(pre, (ny, nx), None)
        assert ret, "Chessboard corners are not found. Please check the number of inside corners in the chessboard."
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)
        objpoints, imgpoints = [], []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        imgp = cv2.cornerSubPix(pre, corners, (11,11), (-1,-1), criteria)
        objpoints = [objp]
        imgpoints = [imgp]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, pre.shape[::-1], None, None)
        h, w = pre.shape
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undist = cv2.undistort(img, mtx, dist, None, newcameramtx)
        newcorners = cv2.undistortPoints(corners, mtx, dist, None, newcameramtx)
        cornerMat = newcorners.reshape(nx, ny, 2).astype(int)
        pts = [cornerMat[0, 0], cornerMat[0, ny-1], cornerMat[nx-1, ny-1], cornerMat[nx-1, 0]]
        return undist, np.array(pts)