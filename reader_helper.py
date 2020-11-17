import cv2
import numpy as np


def detect_staff_lines(gray_img):
    _, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

    staff_img = np.full(binary_img.shape, 255, np.uint8)

    for col_num, col in enumerate(binary_img.T):
        black_spacings = []   # Row differences between black pixels

        last_black = None   # Will have the row of the last black pixel
        for row_num, pixel in enumerate(col):
            if pixel == 0:   # If pixel is black
                if last_black is not None:
                    spacing = row_num - last_black
                    if spacing != 1:
                        black_spacings.append((row_num, spacing))

                last_black = row_num

        counter = 0
        for i, pixel in enumerate(black_spacings):
            row_num, spacing = pixel

            if i+1 < len(black_spacings):
                next_row, next_spacing = black_spacings[i+1]

                if next_spacing in range(spacing-1, spacing+2): # If the next spacing is within 1 of current spacing
                    counter += 1

                    if counter == 3:    # If we found five straight black pixels of (nearly) equal spacing
                        for r in range(row_num - 4*spacing, row_num+spacing, spacing):  # Draw those five pixels
                            staff_img[r, col_num] = 0

                        counter = 0

                else:
                    counter = 0


    cv2.imshow("Staff Image", staff_img)
    cv2.waitKey(0)

