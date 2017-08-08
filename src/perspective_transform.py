import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from threshold import Threshold


class PerpectiveTransform:

    # Static variables
    src = np.float32([[220., 720.], [555., 480.], [730., 480.], [1100., 720.]])
    dst = np.float32([[300, 720], [300, 0], [1280 - 300, 0], [1280 - 300, 720]])

    def __init__(self):
        self.M = None
        self.MInv = None

    def get_perspective_transform_matrix(self):
        """
        :return: Return the perspective transformation matrix
        """
        return self.M

    def get_inverse_perspective_transform_matrix(self):
        """
        :return: Return the perspective transformation matrix
        """
        return self.MInv

    def initialize_perspective_transform_matrix(self):
        """
        Given src and dst points, initialize the perspective transform matrix
        :return:
        """
        self.M = cv2.getPerspectiveTransform(PerpectiveTransform.src,
                                             PerpectiveTransform.dst)

    def initialize_inverse_perspective_transform_matrix(self):
        """
        Given src and dst points, initialize the perspective transform matrix
        :return:
        """
        self.MInv = cv2.getPerspectiveTransform(PerpectiveTransform.dst,
                                                PerpectiveTransform.src)

    def get_four_point_perspective_transform(self, image, M):
        """
        Creates a bird's eye view and returns the transformed image
        and the perspective transformation matrix
        :param image:
        :param: M: perspective transformation matrix
        :return: perspective transformed image
        """

        if M is None:
            print("Perspective transformation matrix is not initialized")
            return

        img = np.copy(image)
        im_shape = (img.shape[1], img.shape[0])

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, im_shape, flags=cv2.INTER_LINEAR)

        return warped

    def get_perspective_transform(self, image):
        if self.M is None:
            self.initialize_perspective_transform_matrix()

        return self.get_four_point_perspective_transform(image, self.M)

    def get_inverse_perspective_transform(self, image):
        if self.MInv is None:
            self.initialize_inverse_perspective_transform_matrix()

        return self.get_four_point_perspective_transform(image, self.MInv)


if __name__=="__main__":

    # Read an image
    img = mpimg.imread('output_images/thresholded_image.jpg')

    # Create a PerpectiveTransform object
    p = PerpectiveTransform()

    # Get the perspective transformed image
    perspective_transposed_image = p.get_perspective_transform(img)
    inverse_perspective_transposed_image = p.get_inverse_perspective_transform(perspective_transposed_image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(perspective_transposed_image)
    ax2.set_title('Perspective Transformed', fontsize=40)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    f.savefig('output_images/perspective_combined.jpg')
    plt.show()