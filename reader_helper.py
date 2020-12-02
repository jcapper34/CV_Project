import math
import cv2
import os
import numpy as np
from music_reader import MEDIA_DIR, TEMPLATE_DIR


class Staff:
    def __init__(self, lines):
        self.lines = lines
        self.clef = None
        self.sharps = None
        self.flats = None

    def __str__(self):
        return str(self.lines)


def detect_staff_lines(binary_img):
    # Thresholds
    MIN_HOUGH_VOTES_FRACTION = 0.6       # threshold = min_hough_votes_fraction * image width
    MIN_LINE_LENGTH_FRACTION = 0.4      # image_width * min_line_length
    MAX_LINE_GAP = 40
    MAX_LINE_ANGLE = 1.0    # Degrees
    LINE_SPACE_VARIATION = 1


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

    lines_img = np.full(binary_img.shape, 255, np.uint8)

    # Filter out lines that arent horizontal
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))
        if abs(line_angle) < MAX_LINE_ANGLE:
            filtered_lines.append([x1, y1, x2, y2])
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)

    filtered_lines.sort(key=lambda x: x[1])     # Sort lines top to bottom

    # Group lines into staffs
    staffs = []
    i = 0
    while i < len(filtered_lines)-1:
        start_i = i
        spacing = filtered_lines[i+1][1] - filtered_lines[i][1]
        while filtered_lines[i+1][1] - filtered_lines[i][1] in range(spacing-LINE_SPACE_VARIATION, spacing+LINE_SPACE_VARIATION+1): # If spacing is about equal
            i += 1
            if i+1 >= len(filtered_lines):
                break
        num_spacings = i - start_i

        assert num_spacings == 4 or num_spacings == 1, "Staff Lines Detected Incorrectly"

        if num_spacings == 4:
            staffs.append(Staff(filtered_lines[start_i: i+1]))

    assert len(staffs) % 2 == 0, "Must have an even number of staffs"

    # cv2.imshow("Staff image", cv2.hconcat([binary_img, lines_img]))
    # cv2.waitKey(0)

    return staffs


#Returns int for bass or trebel clef, trebel = 0, bass = 1
def detect_clefs(bgr_img, staffs):
    C_THRESH = 0.75

    # Read in clef templates
    treble_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'treble-clef.jpg'))
    bass_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'bass-clef.jpg'))

    scale_start, scale_step, scale_stop = 0.1, 0.02, 2.0

    scale = scale_start
    while scale <= scale_stop:
        total_clefs = 0
        found_clefs = []
        scaled_templates = (
            cv2.resize(treble_template, dsize=None, fx=scale, fy=scale),
            cv2.resize(bass_template, dsize=None, fx=scale, fy=scale)
        )
        for scaled_template in scaled_templates:
            c = cv2.matchTemplate(bgr_img, scaled_template, cv2.TM_CCOEFF_NORMED)

            _, binary_scores_img = cv2.threshold(c, thresh=C_THRESH, maxval=255, type=cv2.THRESH_BINARY)
            binary_scores_img = np.array(binary_scores_img, dtype=np.uint8)  # Make sure type is correct

            # Connect the components
            num_clefs, _, _, clef_centroids = cv2.connectedComponentsWithStats(binary_scores_img)

            total_clefs += num_clefs - 1

            found_clefs.append(clef_centroids[1::])
            for x, y in clef_centroids[1::]:  # Only draw if there's a point that meets the threshold
                x, y = int(x), int(y)
                temp_h, temp_w, _ = 50, 50, 0
                cv2.rectangle(bgr_img, (x, y), (x + scaled_template.shape[1], y + scaled_template.shape[0]), (0, 0, 255))

        if total_clefs == len(staffs):
            for i in range(2):
                for x, y in found_clefs[i]:  # Only draw if there's a point that meets the threshold
                    x, y = int(x), int(y)
                    temp_h, temp_w, _ = scaled_templates[i].shape
                    cv2.rectangle(bgr_img, (x, y), (x+temp_w, y+temp_h), (0,0,255))

        scale += scale_step

    cv2.imshow("Cleffs", bgr_img)
    cv2.waitKey(0)


# Note detection
def detect_notes(bgr_img): # NEEDS to take in 1 staff objects at a time, cleff
    # TODO make note object, add note list to staff object
    # Get length and width of staff line
    # Crop this area in sheet music image (bgr_img), slighty larger

    # Perform openings/closings until notes are largest connected components
    # Get coordinates of largest conenected componets (maybe check if they are relatively circular width is about = to height)
    # Order components by x-coord (L -> R)

    # notes = []
    # For every component found
        # Compare area to templates of note values      <-- NOT SURE if we are considering this (worry about the rest first)
        # If no match, not a note
        # Else assign note value

        # Check y coord of center
        # Compare to staff lines y-coords
        # Based on cleff
            # Assign note

        # Add to some array


    # Add note array in

    return ""
