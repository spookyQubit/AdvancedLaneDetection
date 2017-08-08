import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Masking:
    def __init__(self, ver=None):
        self._bottom_x_axis_offset = 120
        self._top_x_axis_offset = 60
        self._top_y_axis_offset = 60
        self._vertices = ver

    def _set_vertices(self, imshape):

        ll = (self._bottom_x_axis_offset, imshape[0])
        tl = (imshape[1] / 2 - self._top_x_axis_offset, imshape[0] / 2 + self._top_y_axis_offset)
        tr = (imshape[1] / 2 + self._top_x_axis_offset, imshape[0] / 2 + self._top_y_axis_offset)
        lr = (imshape[1] - self._bottom_x_axis_offset, imshape[0])

        self._vertices = np.array([[ll, tl, tr, lr]], dtype=np.int32)

    def get_region_of_interest(self, img):
        """
            Applies an image mask.

            Only keeps the region of the image defined by the polygon
            formed from `vertices`. The rest of the image is set to black.
            """
        if self._vertices is None:
            self._set_vertices(img.shape)

        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, self._vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

if __name__=="__main__":

    # Read an image
    img = mpimg.imread('../test_images/test3.jpg')

    # Create a Masking object
    m = Masking()

    # Get an image with the masked region
    masked_image = m.get_region_of_interest(img)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(masked_image)
    ax2.set_title('Masked Image', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()

