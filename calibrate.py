from preprocessing import grayscale
import os, cv2, json, configparser
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
    """
    Get the length of each side.
    Args:
        points (np.ndarray): points of four corners.
    """
    length = []
    for i in range(len(points)):
        length.append(np.linalg.norm(points[i]-points[i-1]).astype(int))
    
    return {"top": length[2], "bottom": length[0], "left": length[1], "right": length[3]}

"""
CameraCalibrator Object
Args:
    preprocessing (callable): a callable function to preprocessing the input image
    cfg_file (str): path to a config file
"""
class CameraCalibrator:
    def __init__(self, preprocessing: callable=grayscale, cfg_file:str=None):
        self.preprocessing = preprocessing
        self.mtx, self.dist, self.newmtx = None, None, None
        if cfg_file is not None:
            self.load(cfg_file)
        
    def calibrate(self, imgs, nx: int, ny: int):
        """
        Calibrate camera
        Args:
            imgs (List[np.ndarray]): input images
            nx (int): number of inside corners in x
            ny (int): number of inside corners in y
        
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints, imgpoints = [], []
        for img in imgs:
            pre = self.preprocessing(img)
            ret, corners = cv2.findChessboardCorners(pre, (ny, nx), None)
            if ret:
                objp = np.zeros((nx * ny, 3), np.float32)
                objp[:,:2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)
                imgp = cv2.cornerSubPix(pre, corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(imgp)
        
        assert len(objpoints) > 0, "Chessboard corners are not found. Please check the number of inside corners in the chessboard."
        
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, pre.shape[::-1], None, None)
        h, w = pre.shape
        self.newmtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
    
    def undistort(self, img):
        """
        Undistort the image after calibration.
        Args:
            imgs (List[np.ndarray]): input images            
            
        Returns:
            an undistorted image
        """
        assert self.mtx is not None, "Please `load` the config file or `calibrate` the camera before undistort."
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.newmtx)
        return undist
    
    def findcorners(self, img, nx: int, ny: int):
        """
        Find corners after calibration.
        Args:
            imgs (List[np.ndarray]): input images
            nx (int): number of inside corners in x
            ny (int): number of inside corners in y
        
        Returns:
            an undistorted image
        """
        assert self.mtx is not None, "Please `load` the config file or `calibrate` the camera before undistort."
        pre = self.preprocessing(img)
        ret, corners = cv2.findChessboardCorners(pre, (ny, nx), None)
        newcorners = cv2.undistortPoints(corners, self.mtx, self.dist, None, self.newmtx)
        cornerMat = newcorners.reshape(nx, ny, 2).astype(int)
        pts = np.array([cornerMat[0, 0], cornerMat[0, ny-1], cornerMat[nx-1, ny-1], cornerMat[nx-1, 0]])
        return pts
    
    def load(self, cfg_file):
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)
        self.mtx = np.array(json.loads(cfg.get("SETTINGS", "mtx")))
        self.dist = np.array(json.loads(cfg.get("SETTINGS", "dist")))
        self.newmtx = np.array(json.loads(cfg.get("SETTINGS", "newmtx")))
    
    def save(self, save_name: str):
        """
        Save config file.
        Args:
            save_name (str): saved name.
        """
        assert self.mtx is not None, "Please `load` the config file or `calibrate` the camera before save configs."
        cfg = configparser.ConfigParser()
        cfg.read(save_name)
        try:
            cfg.add_section("SETTINGS")
        except configparser.DuplicateSectionError:
            pass
        cfg.set("SETTINGS", "mtx", str(self.mtx.tolist()))
        cfg.set("SETTINGS", "dist", str(self.dist.tolist()))
        cfg.set("SETTINGS", "newmtx", str(self.newmtx.tolist()))
        with open(save_name, "w") as f:
            cfg.write(f)