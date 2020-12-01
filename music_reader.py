import os
import cv2
from reader_helper import *

MEDIA_DIR = 'music-img'
TEMPLATE_DIR = 'templates'


def music_reader(filename):
    bgr_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

    # detect_staff_lines(binary_img)

    #If the clef is a trebel clef it returns 0, else `
    clef = detect_clef(bgr_img)



if __name__ == '__main__':
    filename = os.path.join(MEDIA_DIR, 'mary-had-a-little-lamb-bass.png')
    music_reader(filename)
