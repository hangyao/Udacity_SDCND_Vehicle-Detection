from collections import deque
import numpy as np
from scipy.ndimage.measurements import label
from lesson_functions import *

# Define a class to receive the characteristics of each line detection
class Vehicles():
    def __init__(self):
        self.current_bboxes = [] #
        self.recent_bboxes = deque([], 6)
        self.img = None
        self.current_draw_img = None

    def add_bbox_list(self, box_list):
        self.recent_bboxes.append(box_list)
        heat = np.zeros_like(self.img[:,:,0]).astype(np.float)
        for bbox_list in self.recent_bboxes:
            heat = add_heat(heat, bbox_list)
        heat = apply_threshold(heat, len(self.recent_bboxes)/2)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        self.current_draw_img, self.current_bboxes = draw_labeled_bboxes(np.copy(self.img), labels)
        return  self.current_draw_img

    def set_img(self, img):
        self.img = img
