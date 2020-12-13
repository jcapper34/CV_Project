import os
import cv2

from music_playback import *
from reader_helper import *
from image_reader import *

MEDIA_DIR = 'music-img'
TEMPLATE_DIR = 'templates'
REAL_IMG_DIR = 'real-pictures'


def music_reader(filename):
    bgr_img = cv2.imread(filename)
    # bgr_img = find_sheet_music(bgr_img)   # COMMENT OUT to use music from MEDIA_DIR

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, BINARY_THRESH, 255, type=cv2.THRESH_BINARY)  # TODO: could adjust threshold, maybe figure out dynamic threshold approach

    # cv2.imshow("Image", binary_img)
    # cv2.waitKey(0)

    staffs = detect_staffs(binary_img)
    for staff in staffs:
        staff.make_subimage(gray_img)

    markup_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    staffs, markup_image = detect_clefs(staffs, markup_image)

    for staff in staffs:
        markup_image = detect_signature_annotations(staff, markup_image)
        markup_image = detect_notes(staff, markup_image)
        markup_image = detect_rests(staff, markup_image)

    # for clef_num, coord in clef_markup:
    #     x, y = coord
    #     if clef_num == 0:
    #         clef_name = "Treble"
    #     else:
    #         clef_name = "Bass"
    #
    #     markup_image = cv2.putText(markup_image, clef_name, (round(x), round(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255))

    cv2.imshow("Marked-Up Image", markup_image)
    cv2.waitKey(0)

    # audio = create_notes_buffer(staffs[0].notes+staffs[2].notes+staffs[4].notes, play=True)

    play_sheet(staffs, group=2)


if __name__ == '__main__':
    # filename = os.path.join(REAL_IMG_DIR, 'img01.jpg')
    filename = os.path.join(MEDIA_DIR, 'old-macdonald.png')

    music_reader(filename)
