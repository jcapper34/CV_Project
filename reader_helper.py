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

        return ref[int(note_change)]

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
    VERTICAL_LINE_GROUP = 1

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

    # Filter out lines that arent horizontal
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))
        if abs(line_angle) < MAX_LINE_ANGLE:
            filtered_lines.append((x1, y1, x2, y2))

    filtered_lines.sort(key=lambda x: x[1])     # Sort lines top to bottom

    # Merge lines that are directly on top of each other. Accounts for thicker staff lines
    i = 0
    while i < len(filtered_lines):
        line = filtered_lines[i]

        nearby_lines = [line]
        j = i
        while j < len(filtered_lines)-1 and filtered_lines[j+1][1] - line[1] <= VERTICAL_LINE_GROUP:
            line = filtered_lines[j]
            nearby_lines.append(line)
            j += 1

        # Shift line up to make line y value average of vertically connected lines
        y_shift = (sum([l[1] for l in nearby_lines]) / len(nearby_lines)) - line[1]
        filtered_lines[i] = (line[0], line[1]+y_shift, line[2], line[3]+y_shift)

        # Delete the other the vertically connected lines that aren't the original
        for k in range(i, i+len(nearby_lines)-1):
            filtered_lines.pop(k)

        i += 1

    # Draw Staff lines on blank image
    lines_img = np.full(binary_img.shape, 255, np.uint8)
    for x1, y1, x2, y2 in filtered_lines:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)


    cv2.imshow("Staff image", lines_img)
    cv2.waitKey(0)

    # Group lines into staffs
    staffs = []
    i = 0
    while i < len(filtered_lines)-1:
        start_i = i
        spacing = round(filtered_lines[i+1][1] - filtered_lines[i][1])
        while filtered_lines[i+1][1] - filtered_lines[i][1] in range(spacing-LINE_SPACE_VARIATION, spacing+LINE_SPACE_VARIATION+1): # If spacing is about equal
            i += 1
            if i+1 >= len(filtered_lines):
                break
        num_spacings = i - start_i

        assert num_spacings == 4 or num_spacings == 1, "Staff Lines Detected Incorrectly"

        if num_spacings == 4:
            staffs.append(Staff(filtered_lines[start_i: i+1]))

    cv2.imshow("Staff image", lines_img)
    cv2.waitKey(0)

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

    clefs = []
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
                if detected_clef == 0:
                    clef_name = "Treble"
                else:
                    clef_name = "Bass"
                clefs.append((detected_clef, (max_loc[0]+scaled_template.shape[0]+staff.lines[0][0]-60, max_loc[1]+staff_top-10)))

        staff.clef = detected_clef

    return staffs, clefs


def detect_notes(binary_img, staff, annotate=True):
    C_THRESH = 0.7
    vpadding = 0.5
    y_fudge = -1

    # Create subimage
    staff_top = staff.lines[0][1]
    staff_height = staff.lines[-1][1] - staff_top
    staff_image = binary_img[
                  int(staff_top - vpadding * staff_height):int(staff_top + staff_height + vpadding * staff_height),
                  staff.lines[0][0]:staff.lines[0][-2]]  # Create sub-image of staff

    # Create image to show matches
    match_image = cv2.cvtColor(np.copy(staff_image), cv2.COLOR_GRAY2BGR)

    # Get rid of staff lines
    kernel = np.ones((2, 1), np.uint8)
    staff_image = cv2.morphologyEx(staff_image, cv2.MORPH_CLOSE, kernel)

    templates = [   # (Template Image, Counts)
        (cv2.imread(TEMPLATE_DIR+'/quarter-note.jpg'), 1),
        (cv2.imread(TEMPLATE_DIR+'/half-note.jpg'), 2),
        (cv2.imread(TEMPLATE_DIR+'/whole.jpg'), 4)
    ]

    notes_annotations = []  # For drawing an annotated image
    notes = []  # ((x, y), counts)
    for template, counts in templates:
        # Create Binary Template
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # _, template = cv2.threshold(template, 180, 255, cv2.THRESH_BINARY)

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
            match_image = cv2.putText(match_image, letter+str(octave), (x+scaled_template.shape[1], int(y-scaled_template.shape[0]*0.2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (0,0,255))

            # Write number of counts next to note
            match_image = cv2.putText(match_image, str(counts), (x+scaled_template.shape[1], int(y+scaled_template.shape[0]*1.5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (0,0,255))

            if annotate:
                notes_annotations.append((letter+str(octave),(x+scaled_template.shape[1]+staff.lines[0][0], int(y-scaled_template.shape[0]*0.2)+int(staff_top - vpadding * staff_height)),
                                               str(counts), (x+scaled_template.shape[1]+staff.lines[0][0], int(y+scaled_template.shape[0]*1.5)+int(staff_top - vpadding * staff_height))))

    cv2.imshow("Matches", match_image)
    cv2.waitKey(0)

    notes.sort(key=lambda note: note[0][0])     # Sort notes by x value (left to right)

    notes = [(*staff.y_to_note(coord[1]), counts) for coord, counts in notes]   # Notes defined by letter and counts
    staff.notes = notes

    if annotate:
        return notes_annotations
