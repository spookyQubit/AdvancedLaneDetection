from calibration import Camera
from threshold import Threshold
from masking import Masking
from perspective_transform import PerpectiveTransform
from window_search import WindowSearch
from line import Line
from road import Road
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# TODO: Read variables from config


class Process:
    def __init__(self):

        self._image_count = 0

        # Camera variables
        self._calibration_images_dir = '../camera_cal/calibration*.jpg'
        self._calibration_images_nx = 9
        self._calibration_images_ny = 6
        self._calibration_matrix = None
        self._initialize_calibration_matrix()

        # Thresholding variables
        self._sx_params = {"Active": True, "Orient": 'x', "Thresh": (20, 190)}
        self._sy_params = {"Active": True, "Orient": 'y', "Thresh": (20, 190)}
        self._sdir_params = {"Active": True, "Thresh": (0.8, 1.5), "Sobel_kernel": 15}
        self._mag_params = {"Active": True, "Thresh": (0, 255), "Sobel_kernel": 3}
        self._s_color_params = {"Active": True, "Thresh": (170, 255)}
        self._thresholder = None
        self._initialize_thresholder()

        # Masking
        self._masker = None
        self._initialize_masker()

        # Perspective transform
        self._perspective_transformer = None
        self._initialize_perspective_transformer()

        # WindowSearch
        self._window_searcher = None
        self._initialize_window_searcher()

        # Road
        self._road = Road()

    def _initialize_calibration_matrix(self):
        self._calibration_matrix = Camera(self._calibration_images_nx,
                                          self._calibration_images_ny,
                                          self._calibration_images_dir)
        print("Initialized calibration matrix")

    def _initialize_thresholder(self):
        self._thresholder = Threshold(self._sx_params,
                                      self._sy_params,
                                      self._sdir_params,
                                      self._mag_params,
                                      self._s_color_params)
        print("Initialized thresholder")

    def _initialize_masker(self):
        self._masker = Masking()
        print("Initialized masker")

    def _initialize_perspective_transformer(self):
        self._perspective_transformer = PerpectiveTransform()
        print("Initialized perspective transformer")

    def _initialize_window_searcher(self):
        self._window_searcher = WindowSearch()

    def mark_stats(self, image,
                   left_ROC, right_ROC,
                   left_lane_fit, right_lane_fit,
                   lane_width, delta):
        img = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'left_ROC = {:.2f} m'.format(left_ROC), (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'left_lane_fit = [{0:.5f}, {1:.2f}, {2:.2f}]'.format(left_lane_fit[0], left_lane_fit[1], left_lane_fit[2]),
                    (10, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'right_ROC = {:.2f} m'.format(right_ROC), (10, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'right_lane_fit = [{0:.5f}, {1:.2f}, {2:.2f}]'.format(right_lane_fit[0], right_lane_fit[1], right_lane_fit[2]),
                    (10, 160), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'lane_width = {:.2f} m'.format(lane_width), (10, 210), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'delta = {:.2f} m'.format(delta), (10, 240), font, 1, (255, 255, 255), 2,cv2.LINE_AA)
        return img

    def process_image(self, image, mark_stats_on_image=False):
        """

        :param image: An mpimg read image
        :return: Same image with lane superimposed
        """
        img = np.copy(image)

        undistorted_image = self._calibration_matrix.get_undistorted(image)
        img = self._thresholder.apply_threshold(undistorted_image)
        img = self._masker.get_region_of_interest(img)
        img = self._perspective_transformer.get_perspective_transform(img)

        # Get lanes from the image
        current_left_lane, current_right_lane = self._window_searcher.get_lanes(img)
        best_left_ROC, best_right_ROC, best_left_lane_fit, best_right_lane_fit, best_lane_width, best_delta = \
            self._road.get_road_stats(current_left_lane, current_right_lane, img.shape[1], img.shape[0])

        # 1) Make another plot with the lane curves
        # 2) Perform inverse perspective transform on it
        # 3) Stack it on top of the original image
        left_fit = best_left_lane_fit
        right_fit = best_right_lane_fit
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        dst = np.zeros_like(image)
        cv2.fillPoly(dst, np.int_([pts]), (0, 255, 0))

        inv_warp = self._perspective_transformer.get_inverse_perspective_transform(dst)
        dst = cv2.addWeighted(undistorted_image, 1, inv_warp, 0.3, 0)

        self._image_count += 1
        if mark_stats_on_image:
            dst = self.mark_stats(dst,
                                  best_left_ROC, best_right_ROC,
                                  best_left_lane_fit, best_right_lane_fit,
                                  best_lane_width, best_delta)
        return dst

    def process_video(self, input_video, output_video):
        clip = VideoFileClip(input_video)
        clip_with_lane = clip.fl_image(self.process_image)
        clip_with_lane.write_videofile(output_video, audio=False)


def main():
    print("Main")

    process_image = False
    mark_stats_on_image = False
    input_image_file = mpimg.imread('../test_images/test2.jpg')
    output_image_file = 'output_images/road_stats.jpg'

    input_video_file = "../project_video.mp4"
    output_video_file = "project_video_output.mp4"

    p = Process()
    if process_image:
        dst = p.process_image(input_image_file, mark_stats_on_image)
        cv2.imwrite(output_image_file, dst)
        plt.imshow(dst)
        plt.show()
    else:
        p.process_video(input_video_file, output_video_file)


if __name__=="__main__":
    main()