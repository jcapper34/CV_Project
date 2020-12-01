import math
import cv2
import os
import numpy as np
from music_reader import MEDIA_DIR, TEMPLATE_DIR


def detect_staff_lines(binary_img):
    # Thresholds
    MIN_HOUGH_VOTES_FRACTION = 0.6       # threshold = min_hough_votes_fraction * image width
    MIN_LINE_LENGTH_FRACTION = 0.4      # image_width * min_line_length
    MAX_LINE_GAP = 40
    MAX_LINE_ANGLE = 1.0    # Degrees

    lines_img = np.full(binary_img.shape, 255, np.uint8)

    # Find lines with houghLinesP
    image_width = binary_img.shape[1]
    inv_img = 255 - binary_img
    lines = cv2.HoughLinesP(
        image=inv_img,
        rho=1,
        theta=math.pi / 180,
        threshold=int(image_width * MIN_HOUGH_VOTES_FRACTION),
        lines=None,
        minLineLength=int(image_width * MIN_LINE_LENGTH_FRACTION),
        maxLineGap=MAX_LINE_GAP)

    # Filter out lines that arent horizontal
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))
        if abs(line_angle) < MAX_LINE_ANGLE:
            filtered_lines.append([x1, y1, x2, y2])
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)


    cv2.imshow("Staff image", cv2.hconcat([binary_img, lines_img]))
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