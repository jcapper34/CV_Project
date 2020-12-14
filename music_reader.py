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

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, BINARY_THRESH, 255, type=cv2.THRESH_BINARY)  # TODO: could adjust threshold, maybe figure out dynamic threshold approach


    # Make staffs and their subimages
    staffs = detect_staffs(binary_img)
    for staff in staffs:
        staff.make_subimage(gray_img)

    markup_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    staffs, markup_image = detect_clefs(staffs, markup_image)

    for staff in staffs:
        markup_image = detect_signature_accidentals(staff, markup_image)
        markup_image = detect_notes(staff, markup_image)
        markup_image = detect_rests(staff, markup_image)


    cv2.imshow("Annotated Image", markup_image)
    cv2.waitKey(0)

    # Write annotated image to file
    cv2.imwrite(MEDIA_DIR+"/annotated-music.jpg", markup_image)

    # Playback the music
    play_sheet(staffs, group=2)


if __name__ == '__main__':
    filename = os.path.join(MEDIA_DIR, input("Please enter a filename: "))
    # filename = os.path.join(MEDIA_DIR, 'canon-in-d.jpg')

    music_reader(filename)
