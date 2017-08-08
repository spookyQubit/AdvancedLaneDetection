import numpy as np

"""
Purpose:
"""


class Line:
    degree = 2
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self):
        self.ROC = None
        self.fit = None
        pass

    def get_ROC(self, fit, y):
        fit1D = np.poly1d(fit)

        d = fit1D.deriv(1)(y * Line.ym_per_pix)
        dd = fit1D.deriv(2)(y * Line.ym_per_pix)
        r = (1 + d ** 2) ** 1.5 / np.abs(dd)
        return r

    def set_line_stats(self, xy_points, y):
        x_points = xy_points[0]
        y_points = xy_points[1]

        self.fit = np.polyfit(y_points, x_points, Line.degree)
        self.ROC = self.get_ROC(self.fit, y)


if __name__ == "__main__":
    pass
