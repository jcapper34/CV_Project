import os
import cv2

from music_playback import play_sheet
from reader_helper import *
from image_reader import *

MEDIA_DIR = 'music-img'
TEMPLATE_DIR = 'templates'
REAL_IMG_DIR = 'real-pictures'


def music_reader(filename):
    bgr_img = cv2.imread(filename)
    bgr_img = find_sheet_music(bgr_img)   # COMMENT OUT to use music from MEDIA_DIR

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)  # TODO: could adjust threshold, maybe figure out dynamic threshold approach

    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", binary_img)
    # cv2.waitKey(0)

    staffs = detect_staff_lines(binary_img)

    staffs = detect_clefs(binary_img, staffs)

    for staff in staffs:
        detect_notes(gray_img, staff)

    play_sheet(staffs)


if __name__ == '__main__':
    filename = os.path.join(REAL_IMG_DIR, 'twinkle-twinkle-little-star2.jpg')
    #filename = os.path.join(MEDIA_DIR, 'mary-had-a-little-lamb.jpg')
    music_reader(filename)
