import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Threshold:
    def __init__(self,
                 sx_params,
                 sy_params,
                 sdir_params,
                 mag_params,
                 s_color_params):
        self._sx_params = sx_params
        self._sy_params = sy_params
        self._sdir_params = sdir_params
        self._mag_params = mag_params
        self._s_color_params = s_color_params

    def _abs_sobel_thresh(self, img, orient='x', thresh=(0, 255)):

        """
        Input:
        img: An mpimg.imread image
        orient: direction over which the sobel operation needs to be performed (either 'x' or 'y')
        thresh_min: pixel values with less than this will be set to 0
        thresh_min: pixel values with more than this will be set to 0
        Output:
        binary_output: An binary image with
                        shape cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        with non-zero pixel values only
                        where the threshold meets the criteria
        """

        thresh_min = thresh[0]
        thresh_max = thresh[1]

        # Make copy of image
        img = np.copy(img)

        # Convert to Gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Take the derivative in x or y given orient = 'x' or 'y'
        sob = None
        if orient == 'x':
            sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sob)

        # Scale to 8-bit (0 - 255) then convert to type = np.
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1

        # Return this mask as your binary_output image
        return binary_output

    def _mag_sobel_thresh(self, image, thresh=(0, 255), sobel_kernel=3):
        """
        Input:
        img: An mpimg.imread image
        sobel_kernel: An odd number to determine the sobel operation matrix size
        thresh: Threshold to determine the angles which will be picked up
        Output:
        binary_output: An binary image with
                        shape cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        with non-zero pixel values only
                        where the threshold meets the criteria
        """

        img = np.copy(image)

        # Convert to Gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate the magnitude
        abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        sobelScaled = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

        # Create a binary mask where mag thresholds are met
        sobelBinary = np.zeros_like(sobelScaled)
        sobelBinary[(sobelScaled > thresh[0]) & (sobelScaled < thresh[1])] = 1

        return sobelBinary

    def _abs_dir_thresh(self, img, thresh=(0, np.pi / 2), sobel_kernel=15):
        """
        Input:
        img: An mpimg.imread image
        sobel_kernel: An odd number to determine the sobel operation matrix size
        thresh: Threshold to determine the angles which will be picked up
        Output:
        binary_output: An binary image with
                        shape cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        with non-zero pixel values only
                        where the threshold meets the criteria
        """

        # Make copy of image
        img = np.copy(img)

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def _hls_thresh(self, img, thresh=(0, 255)):
        """
        Input:
        img: An mpimg.imread image
        thresh: Threshold to determine the angles which will be picked up
        Output:
        binary_output: An binary image with
                        shape cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        with non-zero pixel values only
                        where the threshold meets the criteria
        """

        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # Apply threshold and create a binary image result
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

        return s_binary

    def apply_threshold(self, image):

        """
        Takes in an image and applies thresholdings as per the params
        :param img:  An mpimg.imread image
        :return: image with all thresholding applied
                 with shape same as image
        """

        sobel_binary_x = None
        sobel_binary_y = None
        sobel_direction = None
        mag_binary = None

        # Make a copy of the image
        img = np.copy(image)

        if img is None:
            return None

        # Sobel x operation
        if self._sx_params["Active"]:
            sobel_binary_x = self._abs_sobel_thresh(img,
                                                    self._sx_params["Orient"],
                                                    self._sx_params["Thresh"])

        # Sobel y operation
        if self._sy_params["Active"]:
            sobel_binary_y = self._abs_sobel_thresh(img,
                                                    self._sy_params["Orient"],
                                                    self._sy_params["Thresh"])

        # Direction gradient
        if self._sdir_params["Active"]:
            sobel_direction = self._abs_dir_thresh(img,
                                                   self._sdir_params["Thresh"],
                                                   self._sdir_params["Sobel_kernel"])

        # Magnitude
        if self._mag_params["Active"]:
            mag_binary = self._abs_dir_thresh(img,
                                              self._mag_params["Thresh"],
                                              self._mag_params["Sobel_kernel"])

        # Color
        if self._s_color_params["Active"]:
            s_color_binary = self._hls_thresh(img,
                                              self._s_color_params["Thresh"])

        if (sobel_binary_x is None) | (sobel_binary_y is None) | (s_color_binary is None):
            return None

        combined = np.zeros((img.shape[0], img.shape[1]))
        combined[((sobel_binary_x == 1) & (sobel_binary_y == 1)) | ((s_color_binary == 1))] = 255
        stacked = np.dstack((combined, combined, combined))

        # Convert it into a 3d image and return
        return stacked


if __name__=="__main__":
    sx_params = {"Active": True, "Orient": 'x', "Thresh": (20, 190)}
    sy_params = {"Active": True, "Orient": 'y', "Thresh": (20, 190)}
    sdir_params = {"Active": False, "Thresh": (0.8, 1.5), "Sobel_kernel": 15}
    mag_params = {"Active": False, "Thresh": (0, 255), "Sobel_kernel": 3}
    s_color_params = {"Active": True, "Thresh": (170, 255)}
    th = Threshold(sx_params, sy_params, sdir_params, mag_params, s_color_params)

    img = mpimg.imread('../test_images/test2.jpg')
    thresholded_image = th.apply_threshold(img)
    cv2.imwrite('output_images/thresholded_image_combined.jpg', np.concatenate((img, thresholded_image), axis=1))
    cv2.imwrite('output_images/thresholded_image.jpg', thresholded_image)

    if thresholded_image is not None:
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=40)

        ax2.imshow(thresholded_image)
        ax2.set_title('Pipeline Result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()
    else:
        print("Thresholding not applied!")
