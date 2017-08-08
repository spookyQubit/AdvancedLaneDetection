import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from line import Line

"""
A class which takes in an image and returns the statistics of the left/right lanes
Statistics:
left_fit
right_fit
left_roc
right_roc
"""

class WindowSearch:
    def __init__(self):
        self._nwindows = 9
        self._margin = 100
        self._minpix = 50
        self.left_fit = None
        self.right_fit = None
        self.left_roc = None
        self.right_roc = None
        self.left_lane = Line()
        self.right_lane = Line()


    def fresh_window_search(self, binary_warped):
        """
        Takes a binary image and retuns the points for left and right lanes
        :param binary_warped: A binary image
        :return: (leftx, lefty), (rightx, righty)
        """

        # Assert that binary_wraped is actually binary
        assert(len(binary_warped.shape) == 2)

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / self._nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self._nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self._margin
            win_xleft_high = leftx_current + self._margin
            win_xright_low = rightx_current - self._margin
            win_xright_high = rightx_current + self._margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self._minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self._minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return (leftx, lefty), (rightx, righty)

    def get_current_fit(self, image):

        # Copy the image
        img = np.copy(image)

        # Keep ony one color and convert it to binary
        if len(img.shape) == 3:
            img = img[:, :, 0]

        # Convert it to binary
        img[img != 0] = 1

        # Get the left/right lane points
        (leftx, lefty), (rightx, righty) = self.fresh_window_search(img)

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return (left_fit, right_fit)

    def plot_lanes(self, img):
        left_fit, right_fit = self.get_current_fit(img)
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)
        plt.show()

    def get_lanes(self, image):
        # Copy the image
        img = np.copy(image)

        # Keep ony one color and convert it to binary
        if len(img.shape) == 3:
            img = img[:, :, 0]

        # Convert it to binary
        img[img != 0] = 1

        # Get left and right lane pixels from image
        (left, right) = self.fresh_window_search(img)
        (leftx, lefty) = (left[0], left[1])
        (rightx, righty) = (right[0], right[1])

        # Get the left/right lane statistics
        self.left_lane.set_line_stats((leftx, lefty), img.shape[0])
        self.right_lane.set_line_stats((rightx, righty), img.shape[0])

        return (self.left_lane, self.right_lane)


if __name__=="__main__":
    # Read an image
    img = mpimg.imread('output_images/perspective_transformed.jpg')

    # Create the object
    w = WindowSearch()
    w.get_lanes(img)
    #w.plot_lanes(img)

    print("main")



