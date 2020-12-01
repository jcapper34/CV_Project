import math
import cv2
import os
import numpy as np
from music_reader import MEDIA_DIR, TEMPLATE_DIR


def detect_staff_lines(binary_img):

    lines_img = np.full(binary_img.shape, 255, np.uint8)
    #
    # for col_num, col in enumerate(binary_img.T):
    #     black_spacings = []   # Row differences between black pixels
    #
    #     last_black = None   # Will have the row of the last black pixel
    #     for row_num, pixel in enumerate(col):
    #         if pixel == 0:   # If pixel is black
    #             if last_black is not None:
    #                 spacing = row_num - last_black
    #                 if spacing != 1:
    #                     black_spacings.append((row_num, spacing))
    #
    #             last_black = row_num
    #
    #     counter = 0
    #     for i, pixel in enumerate(black_spacings):
    #         row_num, spacing = pixel
    #
    #         if i+1 < len(black_spacings):
    #             next_row, next_spacing = black_spacings[i+1]
    #
    #             if next_spacing in range(spacing-1, spacing+2): # If the next spacing is within 1 of current spacing
    #                 counter += 1
    #
    #                 if counter == 3:    # If we found five straight black pixels of (nearly) equal spacing
    #                     for r in range(row_num - 4*spacing, row_num+spacing, spacing):  # Draw those five pixels
    #                         staff_img[r, col_num] = 0
    #
    #                     counter = 0
    #
    #             else:
    #                 counter = 0

    # Find lines with houghLinesP
    image_width = binary_img.shape[1]
    inv_img = 255 - binary_img
    MIN_HOUGH_VOTES_FRACTION = .5       # threshold = min_hough_votes_fraction * image width
    MIN_LINE_LENGTH_FRACTION = .95      # image_width * min_line_length
    houghLines = cv2.HoughLinesP(
        image=inv_img,
        rho=10,
        theta=math.pi / 180,
        threshold=int(image_width * MIN_HOUGH_VOTES_FRACTION),
        lines=None,
        minLineLength=int(image_width * MIN_LINE_LENGTH_FRACTION),
        maxLineGap=10)

    # Account for same lines counted multiple times
    seen_lines = []
    for i in range(0, len(houghLines)):
        seen_lines.append(houghLines[i][0][1])

    for i in range(0, len(seen_lines)):
        for j in range(i + 1, len(seen_lines)):
            diff = abs(seen_lines[i] - seen_lines[j])
            if(diff < 5):
                seen_lines.pop(i)
                break

    # For visualizing the lines
    for k in range(0, len(houghLines)):
        if houghLines[k][0][1] in seen_lines:
            l = houghLines[k][0]
            cv2.line(lines_img, (l[0], l[1]), (l[2], l[3]), 0,
                     thickness=2, lineType=cv2.LINE_AA)
    cv2.namedWindow("Staff image", cv2.WINDOW_NORMAL)
    cv2.imshow("Staff image", lines_img)
    cv2.waitKey(0)

    # TODO return a set of the hough lines whose y is in seen_lines
    # figure out best way to store these ^ so we can scan across lines, check y's of notes relative to lines


#Returns int for bass or trebel clef, trebel = 0, bass = 1
def detect_clef(bgr_img):
    # Treble clef matching
    treble_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'treble-clef.jpg'))

    c = cv2.matchTemplate(bgr_img, treble_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val_trebel, min_loc, max_loc = cv2.minMaxLoc(c)
    x, y = max_loc

    threshold = 0.5
    if max_val_trebel > threshold:  # Only draw if there's a point that meets the threshold
        cv2.rectangle(bgr_img, (x, y), (x + 20, y + 20), (0, 0, 255),
                      2)  # Draw Rectangle from top left corner

    # Bass clef matching
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

# Note detection
def detect_note(bgr_img):

    return ""