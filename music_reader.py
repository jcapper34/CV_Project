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
    _, binary_img = cv2.threshold(gray_img, BINARY_THRESH, 255, cv2.THRESH_BINARY)  # TODO: could adjust threshold, maybe figure out dynamic threshold approach

    # cv2.imshow("Image", binary_img)
    # cv2.waitKey(0)

    staffs = detect_staff_lines(binary_img)
    for staff in staffs:
        staff.make_subimage(gray_img)

    staffs, clefs = detect_clefs(staffs)

    # Draw an annotated image for debugging
    annotated_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for staff in staffs:
        detect_rests(staff)
        for s1, coord1, s2, coord2 in detect_notes(staff):
            annotated_img = cv2.putText(annotated_img, s1, coord1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255))
            annotated_img = cv2.putText(annotated_img, s2, coord2, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255))

    for clef_num, coord in clefs:
        x, y = coord
        if clef_num == 0:
            clef_name = "Treble"
        else:
            clef_name = "Bass"

        annotated_img = cv2.putText(annotated_img, clef_name, (round(x), round(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255))

    cv2.imshow("Annotated Image", annotated_img)

    # audio = create_notes_buffer(staffs[1].notes+staffs[3].notes+staffs[5].notes, play=True)
    play_sheet(staffs[:2])

    cv2.waitKey(0)


if __name__ == '__main__':
    # filename = os.path.join(REAL_IMG_DIR, 'twinkle-twinkle-little-star2.jpg')
    filename = os.path.join(MEDIA_DIR, 'old-macdonald.png')
    music_reader(filename)

