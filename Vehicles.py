from collections import deque
import numpy as np
from scipy.ndimage.measurements import label
from helper_functions import *

# Define a class to receive the characteristics of each line detection
class Vehicles():
    def __init__(self):
        self.current_bboxes = []
        self.recent_bboxes = deque([], 12)
        self.img = None
        self.draw_img = None
        self.frame = 0
        self.new_ystop = 0

    def draw_over_frames(self, box_list):
        self.recent_bboxes.append(box_list)
        heat = np.zeros_like(self.img[:,:,0]).astype(np.float)
        for bbox_list in self.recent_bboxes:
            heat = add_heat(heat, bbox_list)
        heat = apply_threshold(heat, len(self.recent_bboxes)/3)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        self.draw_img, car_box = draw_labeled_bboxes(np.copy(self.img), labels)
        if np.array(car_box).any():
            self.current_bboxes = car_box
            self.new_ystop = np.amax(np.array(car_box), axis=0)[1,1] + 64
        else:
            self.new_ystop = 0
        return  self.draw_img
