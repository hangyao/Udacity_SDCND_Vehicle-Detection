import numpy as np
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque([], 6)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        #self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def update_line(self, x, y, fitparam, xfit, curv, dist):
        self.detected = True
        self.allx = x
        self.ally = y
        self.current_fit = fitparam
        self.recent_xfitted.append(xfit)
        self.radius_of_curvature = curv
        self.line_base_pos = dist
        self.bestx = np.mean(self.recent_xfitted, axis=0)
