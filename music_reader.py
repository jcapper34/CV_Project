import os
import cv2

MEDIA_DIR = 'music-img'


def music_reader(filename):
    bgr_img = cv2.imread(filename)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Image", gray_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    filename = os.path.join(MEDIA_DIR, 'hot-cross-buns.png')
    music_reader(filename)
