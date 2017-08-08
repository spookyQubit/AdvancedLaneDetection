from line import Line
import numpy as np

class Road:
    """
    This is a stateful class which keeps the memory of
    the previously detected lanes
    """
    max_width = 4.5 # in meters
    min_width = 3.1 # in meters
    min_roc = 800 # in meters

    def __init__(self):
        self.best_left_ROC = None
        self.best_right_ROC = None
        self.best_left_lane_fit = None
        self.best_right_lane_fit = None
        self.best_lane_width = None
        self.current_lane_width = None
        self.best_delta = None
        self.num_images_accepted = 0
        self.num_images_rejected = 0
        self.current_ratio = 0.80
        self.previous_ratio = 1 - self.current_ratio

    def _initialize_params(self, current_left_lane, current_right_lane, x, y):

        if self.best_left_ROC is None:
            self.best_left_ROC = current_left_lane.ROC

        if self.best_right_ROC is None:
            self.best_right_ROC = current_right_lane.ROC

        if self.best_left_lane_fit is None:
            self.best_left_lane_fit = current_left_lane.fit

        if self.best_right_lane_fit is None:
            self.best_right_lane_fit = current_right_lane.fit

        if self.best_lane_width is None:
            self.best_lane_width = self.get_lane_width(current_left_lane, current_right_lane, y)

        if self.best_delta is None:
            self.best_delta = self.get_current_delta(current_left_lane, current_right_lane, x, y)

    def get_road_stats(self, current_left_lane, current_right_lane, xmax, ymax):
        """
        Main entry point of the code
        :param current_left_lane: Line fit
        :param current_right_lane: Line fit
        :param xmax: xmax, ie img.shape[1]
        :param ymax: ymax, ie img.shape[1]
        :return: lane stats
        """
        self._initialize_params(current_left_lane, current_right_lane, xmax, ymax)

        if self.is_sane(current_left_lane, current_right_lane, xmax, ymax):
            self.best_left_ROC = self.current_ratio * current_left_lane.ROC + self.previous_ratio * self.best_left_ROC
            self.best_right_ROC = self.current_ratio * current_right_lane.ROC + self.previous_ratio * self.best_right_ROC
            self.best_left_lane_fit = self.current_ratio * current_left_lane.fit + self.previous_ratio * self.best_left_lane_fit
            self.best_right_lane_fit = self.current_ratio * current_right_lane.fit + self.previous_ratio * self.best_right_lane_fit
            self.best_lane_width = self.current_ratio * self.get_average_lane_width(current_left_lane, current_right_lane, ymax)[1] \
                                   + self.previous_ratio * self.best_lane_width
            self.best_delta = self.current_ratio * self.get_current_delta(current_left_lane, current_right_lane, xmax, ymax) \
                              + self.previous_ratio * self.best_delta
            self.num_images_accepted += 1
        else:
            self.num_images_rejected += 1
            print("number of images accepted = ", self.num_images_accepted)
            print("number of images rejected = ", self.num_images_rejected)

        return self.best_left_ROC, self.best_right_ROC, \
               self.best_left_lane_fit, self.best_right_lane_fit, \
               self.best_lane_width, self.best_delta

    def get_current_delta(self, current_left_lane, current_right_lane, x, y):
        left_fit = current_left_lane.fit
        right_fit = current_right_lane.fit

        x_left = (left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]) * Line.xm_per_pix
        x_right = (right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]) * Line.xm_per_pix

        return np.absolute(x * Line.xm_per_pix / 2.0 - (x_left + x_right) / 2.0)

    def get_lane_width(self, current_left_lane, current_right_lane, y):
        """
        Given the current left and right lanes, check:
        calculates the lane width and returns the width in meters
        :param current_left_lane: Line instance
        :param current_right_lane: Line instance
        :return: Lane width in meters
        """
        left_fit = current_left_lane.fit
        right_fit = current_right_lane.fit

        x_left = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
        x_right = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

        width = (x_right - x_left) * Line.xm_per_pix
        return width


    def get_average_lane_width(self, current_left_lane, current_right_lane, ymax):
        widths = []

        left_fit = current_left_lane.fit
        right_fit = current_right_lane.fit

        num_points = 10
        ymin = 0
        for y in range(ymin, ymax, int((ymax - ymin) / num_points)):
            w = self.get_lane_width(current_left_lane, current_right_lane, y)
            widths.append(w)
            #print("y ={}, width = {}".format(y, w))

        if (min(widths) < Road.min_width) or (max(widths) > Road.max_width):
            return False, -1

        return True, np.mean(widths)

    def is_sane(self, current_left_lane, current_right_lane, xmax, ymax):
        """
        Given the current left and right lanes, check:
        1) width of the lanes is reasonable
        2) The lanes are nearly parallel
        3) The roc is reasonable
        :param current_left_lane: Line instance
        :param current_right_lane: Line instance
        :return: True/False
        """

        is_width_correct, current_width = self.get_average_lane_width(current_left_lane, current_right_lane, ymax)
        current_left_roc = current_left_lane.ROC
        current_right_roc = current_right_lane.ROC

        if not is_width_correct:
            print("current_width = ", current_width)
            return False

        if (current_left_roc < Road.min_roc) or (current_right_roc < Road.min_roc):
            print("current_left_roc = ", current_left_roc)
            print("current_right_roc = ", current_right_roc)
            return False

        return True

