import os
import cv2
from reader_helper import *

MEDIA_DIR = 'music-img'


def music_reader(filename):
    bgr_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    detect_staff_lines(gray_img)


if __name__ == '__main__':
    filename = os.path.join(MEDIA_DIR, 'hot-cross-buns.png')
    music_reader(filename)
