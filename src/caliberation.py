import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Camera:

    def __init__(self, nx, ny, calibrationImagesDir):
        self._nx = nx
        self._ny = ny
        self._calibrationImagesDir = calibrationImagesDir
        self._mtx = None
        self._dist = None
        self._ret = False
        self._imagePoints = None
        self._objPoints = None

    def _calibrate(self, imageShape):
        """
        Sets the _mtx, _dist, _ret by using cv2.calibrateCamera
        :param imageShape: An array of [height, width, <channels>]
        :return: void
        """
        # Read the calibration image
        images = glob.glob(self._calibrationImagesDir)

        # Array to store object points and image points
        objpoints = []
        imgpoints = []

        objp = np.zeros((self._nx * self._ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self._nx, 0:self._ny].T.reshape(-1, 2)

        for fname in images:
            img_ = mpimg.imread(fname)
            img = np.copy(img_)

            # Convert to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chess-board corners
            ret, corners = cv2.findChessboardCorners(gray, (self._nx, self._ny), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Get calibration objects
        self._ret, self._mtx, self._dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints,
                                imgpoints,
                                imageShape[0:2],
                                None, None)

    def get_undistorted(self, img):
        """
        Returns the undistorted image using the calibration matrix
        :param img: An mpimg.imread image which we want to undistort
        :return: undistorted image of the same shape as input img
        """
        undist = None
        if not self._ret:
            self._calibrate(img.shape)

        if self._ret:
            undist = cv2.undistort(img, self._mtx, self._dist, None, self._mtx)
        else:
            print("Camera: Cannot calibrate and undistort")

        return undist


if __name__=="__main__":

    glob_directory = '../camera_cal/calibration*.jpg'
    nx = 9
    ny = 6

    c = Camera(nx, ny, glob_directory)

    img = mpimg.imread('../camera_cal/calibration1.jpg')
    undist = c.get_undistorted(img)

    if undist is not None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()

        # Plot the distorted image
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=40)

        # Plot the undistorted image
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=40)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    else:
        print("Undistroted image is None")

