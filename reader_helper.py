import cv2
import numpy as np


def detect_staff_lines(gray_img):
    _, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

    staff_img = np.full(binary_img.shape, 255, np.uint8)

    for col_num, col in enumerate(binary_img.T):
        row_d = []   # Row differences between black pixels
        last_black = None   # Will have the row of the last black pixel
        for row_num, pixel in enumerate(col):
            if pixel == 0:   # If pixel is black
                if last_black is not None:
                    difference = row_num - last_black
                    if difference != 1:
                        row_d.append(row_num - last_black)
                        staff_img[row_num, col_num] = 0

                last_black = row_num


    cv2.imshow("Staff Image", staff_img)
    cv2.waitKey(0)

