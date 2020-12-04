from pprint import pprint
import math
import cv2
import os
import numpy as np
from music_reader import MEDIA_DIR, TEMPLATE_DIR


class Staff:
    def __init__(self, lines):
        self.lines = lines
        self.notes = None
        self.clef = None    # 0 if treble else bass if 1
        self.sharps = None
        self.flats = None
        self.img = None

    # Super sloppy way of determining the note from y coordinate. TODO: Change before final report
    def y_to_note(self, y):
        letter_range_start = ord('A')
        letter_range_end = ord('G')

        if self.clef == 0:
            ref = [('F', 5), ('E', 5), ('D', 5), ('C', 5), ('B', 4), ('A', 4), ('G', 4), ('F', 4), ('E', 4), ('D', 4),
                   ('C', 4)]
        else:
            ref = [('A', 3), ('G', 3), ('F', 3), ('E', 3), ('D', 3), ('C', 3), ('B', 2), ('A', 2), ('G', 2), ('F', 2),
                   ('E', 2)]


        staff_top = self.lines[0][1]
        note_spacing = (self.lines[-1][1] - staff_top) / ((len(self.lines)-1)*2)  # Vertical note spacing (half of line spacing)

        note_change = round((y-staff_top) / note_spacing)

        return ref[note_change]

    def __str__(self):
        return ("Staff:\n" +
        "\tlines: %s\n" +
        "\tclef: %s\n" +
        "\tnotes: %s") % (str(self.lines),
                         "Treble" if self.clef == 0 else "Bass" if self.clef == 1 else 'None',
                          str(self.notes))


def detect_staff_lines(binary_img):
    # Thresholds
    MIN_HOUGH_VOTES_FRACTION = 0.7       # threshold = min_hough_votes_fraction * image width
    MIN_LINE_LENGTH_FRACTION = 0.4      # image_width * min_line_length
    MAX_LINE_GAP = 40
    MAX_LINE_ANGLE = 1.0    # Degrees
    LINE_SPACE_VARIATION = 1


    # TODO: Houghlines for thick staff lines
    # Find black lines with houghLinesP
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
    # cv2.imshow("Staff image", lines_img)
    # cv2.waitKey(0)

    return staffs


def detect_clefs(binary_img, staffs):
    C_THRESH = 0.5        # Template matching threshold
    TREBLE_PADDING = 0.5   # The treble clef extends from around 50% above to 50% below the staff

    # Read in clef templates as binary images
    treble_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'treble-clef.jpg'))
    treble_template = cv2.cvtColor(treble_template, cv2.COLOR_BGR2GRAY)
    _, treble_template = cv2.threshold(treble_template, 127, 255, cv2.THRESH_BINARY)

    bass_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'bass-clef.jpg'))
    bass_template = cv2.cvtColor(bass_template, cv2.COLOR_BGR2GRAY)
    _, bass_template = cv2.threshold(bass_template, 127, 255, cv2.THRESH_BINARY)

    for staff in staffs:
        staff_top = staff.lines[0][1]
        staff_height = staff.lines[-1][1] - staff_top
        staff_image = binary_img[int(staff_top - TREBLE_PADDING * staff_height):int(staff_top + staff_height + TREBLE_PADDING * staff_height),
                            staff.lines[0][0]:staff.lines[0][-2]]   # Create sub-image of staff

        detected_clef = None
        # TODO: Slightly vary template scale until a sweet spot is found
        for clef_num, template in enumerate((treble_template, bass_template)):
            scale = staff_image.shape[0] / template.shape[0]
            scaled_template = cv2.resize(template, dsize=None, fx=scale, fy=scale)  # Scale template to be same height as staff

            c = cv2.matchTemplate(staff_image, scaled_template, cv2.TM_CCOEFF_NORMED)   # Get template matching scores
            _, max_val, _, max_loc = cv2.minMaxLoc(c)

            if max_val > C_THRESH:
                detected_clef = clef_num

        staff.clef = detected_clef

        # cv2.imshow("Staff", staff_image)
        # cv2.waitKey(0)

    return staffs


# def detect_notes(gray_img, staff): # NEEDS to take in 1 staff objects at a time, cleff
#     # Grabs the position of the lines in the staff and adds a buffer
#     y1 = staff.lines[0][1]
#     y5 = staff.lines[4][1]
#     buffer_dif_y = int((y5 - y1))
#     x_start = staff.lines[0][0]
#     x_stop = staff.lines[0][2]
#     buffer_dif_x = int((x_stop - x_start) / 15)
#     crop_img = gray_img[y1 - buffer_dif_y:y5 + buffer_dif_y, x_start + buffer_dif_x:x_stop]
#     staff.img = crop_img
#     # Crop this area in sheet music image (bgr_img), slighty larger
#
#     # Showing for testing
#     cv2.imshow("Cropped", crop_img)
#     # cv2.waitKey(0)
#
#     # Get the binary image
#     _, thresh_img = cv2.threshold(crop_img, 240, 255, cv2.THRESH_BINARY)  # Get rid of gray values (helps fill in notes)
#
#     # Perform openings/closings until notes are largest connected components
#     kernel = np.ones((2, 1), np.uint8)
#     filtered_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)  # Get rid of staff lines
#     kernel = np.ones((1, 4), np.uint8)
#     filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE,
#                                     kernel)  # Get rid of horizontal lines (for eight and sixteenth notes)
#     kernel = np.ones((5, 3), np.uint8)
#     filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)  # Fill in half and whole notes
#     kernel = np.ones((5, 1), np.uint8)
#     filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel)  # Get rid of other horizontal lines
#     kernel = np.ones((1, 4), np.uint8)
#     filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel)  # Get rid of other vertical lines
#     cv2.imshow("Filtered", filtered_img)  # TESTING
#     cv2.waitKey(0)
#     # Get coordinates of largest conenected componets (maybe check if they are relatively circular width is about = to height)
#     # Order components by x-coord (L -> R)
#
#     # notes = []
#     # For every component found
#         # Compare area to templates of note values      <-- NOT SURE if we are considering this (worry about the rest first)
#         # If no match, not a note
#         # Else assign note value
#
#         # Check y coord of center
#         # Compare to staff lines y-coords
#         # Based on cleff
#             # Assign note
#
#         # Add to some array
#
#
#     # Add note array in
#
#     # padding = 0.5
#     # for staff in staffs:
#     #     staff_top = staff.lines[0][1]
#     #     staff_height = staff.lines[-1][1] - staff_top
#     #     staff_image = binary_img[int(staff_top - padding * staff_height):int(staff_top + staff_height + padding * staff_height),
#     #                   staff.lines[0][0]:staff.lines[0][-2]]  # Create sub-image of staff
#
#
#     return ""


def detect_notes(gray_img, staff):
    C_THRESH = 0.7
    vpadding = 0.5
    y_fudge = -1

    # Create subimage
    staff_top = staff.lines[0][1]
    staff_height = staff.lines[-1][1] - staff_top
    staff_image = gray_img[
                  int(staff_top - vpadding * staff_height):int(staff_top + staff_height + vpadding * staff_height),
                  staff.lines[0][0]:staff.lines[0][-2]]  # Create sub-image of staff

    # Create image to show matches
    match_image = cv2.cvtColor(np.copy(staff_image), cv2.COLOR_GRAY2BGR)

    # Get rid of staff lines
    kernel = np.ones((2, 1), np.uint8)
    staff_image = cv2.morphologyEx(staff_image, cv2.MORPH_CLOSE, kernel)

    templates = [   # (Template Image, Counts)
        (cv2.imread(TEMPLATE_DIR+'/quarter-note.jpg'), 1),
        (cv2.imread(TEMPLATE_DIR+'/half-note.jpg'), 2)
    ]

    notes = []  # ((x, y), counts)
    for template, counts in templates:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)   # Grayscale the template

        # Scale template
        scale = (staff_height/4) / template.shape[0]
        scaled_template = cv2.resize(template, dsize=None, fx=scale, fy=scale)

        c = cv2.matchTemplate(staff_image, scaled_template, cv2.TM_CCOEFF_NORMED)

        # Get all points at threshold
        _, thresh_img = cv2.threshold(c, thresh=C_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        thresh_img = np.array(thresh_img, dtype=np.uint8)  # Make sure type is correct

        # Close note blobs
        kernel = np.ones(scaled_template.shape, np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("Thresh", thresh_img)
        # cv2.waitKey(0)

        _, _, _, centroids = cv2.connectedComponentsWithStats(thresh_img)

        for x, y in centroids[1::]:
            y += y_fudge    # Lazy shift to fix issue. TODO: Change before final report

            # Get note center
            cx = x + scaled_template.shape[1]/2
            cy = y + scaled_template.shape[0]/2 + int(staff_top - vpadding * staff_height)
            notes.append(((cx, cy), counts))

            letter, octave = staff.y_to_note(cy)

            x = int(x)
            y = int(y)

            # Draw Rectangle around note
            # cv2.rectangle(match_image, (x, y), (x + scaled_template.shape[1], y + scaled_template.shape[0]), (0, 0, 255), 1)

            font_scale = 0.6
            # Write note value next to note
            match_image = cv2.putText(match_image, letter+str(octave), (x+scaled_template.shape[1], y), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (0,0,255))

            # Write number of counts next to note
            match_image = cv2.putText(match_image, str(counts), (x+scaled_template.shape[1], int(y+scaled_template.shape[0]*1.4)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (0,0,255))

    cv2.imshow("Matches", match_image)
    cv2.waitKey(0)

    notes.sort(key=lambda note: note[0][0])     # Sort notes by x value (left to right)

    notes = [(*staff.y_to_note(coord[1]), counts) for coord, counts in notes]   # Notes defined by letter and counts
    staff.notes = notes


