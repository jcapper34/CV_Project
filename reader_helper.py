import math
import cv2
import os
import numpy as np
from music_reader import MEDIA_DIR, TEMPLATE_DIR


class Staff:
    def __init__(self, lines):
        self.lines = lines
        self.clef = None    # 0 if treble else bass if 1
        self.sharps = None
        self.flats = None

    def __str__(self):
        return ("Staff:\n" +
        "\tlines: %s\n" +
        "\tclef: %s") % (str(self.lines), str(self.clef))


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
    C_THRESH = 0.6         # Template matching threshold
    TREBLE_PADDING = 0.5   # The treble clef extends from around 50% above to 50% below the staff

    # Read in clef templates
    treble_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'treble-clef.jpg'))
    bass_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'bass-clef.jpg'))

    for staff in staffs:
        staff_top = staff.lines[0][1]
        staff_height = staff.lines[-1][1] - staff_top
        staff_image = bgr_img[int(staff_top-TREBLE_PADDING*staff_height):int(staff_top+staff_height+TREBLE_PADDING*staff_height),
                            staff.lines[0][0]:staff.lines[0][-2]]   # Create sub-image of staff

        detected_clef = None
        for clef_num, template in enumerate((treble_template, bass_template)):
            scale = staff_image.shape[0] / template.shape[0]
            scaled_template = cv2.resize(template, dsize=None, fx=scale, fy=scale)  # Scale template to be same height as staff

            c = cv2.matchTemplate(staff_image, scaled_template, cv2.TM_CCOEFF_NORMED)   # Get template matching scores
            _, max_val, _, max_loc = cv2.minMaxLoc(c)

            if max_val > C_THRESH:
                detected_clef = clef_num

        staff.clef = detected_clef

    return staffs


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
