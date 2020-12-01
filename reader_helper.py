import cv2
import os
import numpy as np
from music_reader import MEDIA_DIR, TEMPLATE_DIR


def detect_staff_lines(binary_img):

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


#Returns int for bass or trebel clef, trebel = 0, bass = 1
def detect_clef(bgr_img):
    treble_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'treble-clef.jpg'))

    c = cv2.matchTemplate(bgr_img, treble_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val_trebel, min_loc, max_loc = cv2.minMaxLoc(c)
    x, y = max_loc

    threshold = 0.5
    if max_val_trebel > threshold:  # Only draw if there's a point that meets the threshold
        cv2.rectangle(bgr_img, (x, y), (x + 20, y + 20), (0, 0, 255),
                      2)  # Draw Rectangle from top left corner
    bass_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'bass-clef.jpg'))

    c = cv2.matchTemplate(bgr_img, bass_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val_bass, min_loc, max_loc = cv2.minMaxLoc(c)
    x, y = max_loc

    threshold = 0.5
    if max_val_bass > threshold:  # Only draw if there's a point that meets the threshold
        cv2.rectangle(bgr_img, (x, y), (x + 20, y + 20), (0, 0, 255),
                      2)  # Draw Rectangle from top left corner
    if(max_val_trebel > max_val_bass):
        return 0
    else:
        return 1
