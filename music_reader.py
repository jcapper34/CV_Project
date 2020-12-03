import os
import cv2
from reader_helper import *

MEDIA_DIR = 'music-img'
TEMPLATE_DIR = 'templates'


def music_reader(filename):
    bgr_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

    staffs = detect_staff_lines(binary_img)

    #If the clef is a trebel clef it returns 0, else if it is a bass clef return 1
    staffs = detect_clefs(binary_img, staffs)

    for staff in staffs:
        detect_notes(bgr_img, staff)


if __name__ == '__main__':
    filename = os.path.join(MEDIA_DIR, 'mary-had-a-little-lamb.jpg')
    music_reader(filename)


    # bass_template = cv2.imread("templates/bass-clef.JPG")
    # row = np.array([(255, 255, 255) for j in range(bass_template.shape[1])])
    #
    # for i in range(46):
    #     bass_template = np.insert(bass_template, 0, row, axis=0)
    #
    # for i in range(45):
    #
    #     bass_template = np.insert(bass_template, -1, row, axis=0)
    #
    #
    # cv2.imwrite("templates/bass-clef.JPG", bass_template)
