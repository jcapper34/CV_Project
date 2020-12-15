import math
import cv2
import os
import numpy as np
from music_reader import MEDIA_DIR, TEMPLATE_DIR

BINARY_THRESH = 180

# Template Matching Thresholds
CLEF_THRESH = 0.55
NOTE_THRESH = 0.7
REST_THRESH = 0.7
ACCIDENTAL_THRESH = 0.6

# Divide the width of image by this to get a nice font size for the markup image
MARKUP_FONT_DIVIDER = 1600


class Staff:
    vpadding = 0.5

    def __init__(self, lines):
        self.lines = lines
        self.notes = []             # (value(letter, octave, counts), coordinates(x, y)). If rest then value=counts

        self.clef = None            # 0 if treble else bass if 1
        self.clef_dim = None        # Dimensions of clef so it can be cropped out
        self.clef_loc = None        # Location of clef (relative to staff image) so it can be cropped out

        # Signature accidentals
        self.sharps = []
        self.flats = []

        self.gray_img = None        # Grayscale subimage of d
        self.binary_img = None

    def make_subimage(self, gray_img):
        staff_top = self.lines[0][1]
        staff_height = self.lines[-1][1] - staff_top
        self.gray_img = gray_img[round(staff_top - self.vpadding * staff_height):round(
            staff_top + staff_height + self.vpadding * staff_height),
                      self.lines[0][0]:self.lines[0][-2]]  # Create sub-image of staff

        _, self.binary_img = cv2.threshold(self.gray_img, BINARY_THRESH, 255, cv2.THRESH_OTSU)

    # Super sloppy way of determining the note from y coordinate. TODO: Change before final report
    def classify_note(self, y):
        letter_range_start = ord('A')
        letter_range_end = ord('G')

        if self.clef == 0:
            ref = [('F', 5), ('E', 5), ('D', 5), ('C', 5), ('B', 4), ('A', 4), ('G', 4), ('F', 4), ('E', 4), ('D', 4),
                   ('C', 4), ('B', 3), ('A', 3)]
        else:
            ref = [('A', 3), ('G', 3), ('F', 3), ('E', 3), ('D', 3), ('C', 3), ('B', 2), ('A', 2), ('G', 2), ('F', 2),
                   ('E', 2), ('D', 2), ('C', 2), ('B', 1)]

        staff_top = self.lines[0][1]
        note_spacing = (self.lines[-1][1] - staff_top) / ((len(self.lines)-1)*2)  # Vertical note spacing (half of line spacing)

        note_change = round((y-staff_top) / note_spacing)

        letter, octave = ref[int(note_change)]

        accidental = None
        if letter in self.sharps:
            accidental = '#'
        elif letter in self.flats:
            accidental = 'b'

        return letter, accidental, octave

    def __str__(self):
        return ("Staff:\n" +
        "\tlines: %s\n" +
        "\tclef: %s\n" +
        "\tnotes: %s") % (str(self.lines),
                         "Treble" if self.clef == 0 else "Bass" if self.clef == 1 else 'None',
                          str(self.notes))


# Uses Hough lines to detect staff lines and return list of staff objects
def detect_staffs(binary_img):
    # Thresholds
    MIN_HOUGH_VOTES_FRACTION = 0.7       # threshold = min_hough_votes_fraction * image width
    MIN_LINE_LENGTH_FRACTION = 0.5      # image_width * min_line_length
    MAX_LINE_GAP = 20
    MAX_LINE_ANGLE = 1.0                    # Degrees from horizontal
    LINE_SPACE_VARIATION = 1/3             # The percent error allowed in vertical spacing when grouping into staffs
    VERTICAL_LINE_GROUP = 3                 # The vertical difference between two lines that will make them be treated as the same line

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

    # Group lines into staffs
    staffs = []
    i = 0
    while i < len(filtered_lines)-1:
        start_i = i
        ref_spacing = round(filtered_lines[i+1][1] - filtered_lines[i][1])  # Spacing between adjacent lines

        # While the next spacings are within LINE_SPACE_VARIATION of the reference spacing
        while filtered_lines[i+1][1] - filtered_lines[i][1] in range(int(ref_spacing*(1-LINE_SPACE_VARIATION)), int(ref_spacing*(1+LINE_SPACE_VARIATION))): # If spacing is about equal
            i += 1
            if i+1 >= len(filtered_lines):
                break
        num_spacings = i - start_i

        # If five consecutive lines have similar spacings
        if num_spacings == 4:
            staffs.append(Staff(filtered_lines[start_i:i+1]))

    cv2.imshow("Staff image", cv2.hconcat([binary_img, lines_img]))
    cv2.waitKey(0)
    return staffs


# Finds the clef for each staff in a list. Sets clef variable for each staff and returns marked up image
def detect_clefs(staffs, markup_image):
    tune_start, tune_stop, tune_step = 0.9, 1, 0.02     # Scale tuning variables

    # Read in clef templates as gray images
    treble_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'treble-clef.jpg'))
    treble_template = cv2.cvtColor(treble_template, cv2.COLOR_BGR2GRAY)

    bass_template = cv2.imread(os.path.join(TEMPLATE_DIR, 'bass-clef.jpg'))
    bass_template = cv2.cvtColor(bass_template, cv2.COLOR_BGR2GRAY)

    for staff in staffs:
        staff_top = staff.lines[0][1]
        staff_height = staff.lines[-1][1] - staff_top

        detected_clef = None
        detected_clef_dim = None
        detected_clef_loc = None

        # Try to match each clef template
        for clef_num, template in enumerate((treble_template, bass_template)):
            # Slightly tune template scaling until sweet spot is found
            tune = tune_start
            while tune < tune_stop:
                scale = staff.binary_img.shape[0] / template.shape[0]
                scale *= tune

                scaled_template = cv2.resize(template, dsize=None, fx=scale, fy=scale)  # Scale template to be same height as staff image

                c = cv2.matchTemplate(staff.gray_img, scaled_template, cv2.TM_CCOEFF_NORMED)   # Get template matching scores
                _, max_val, _, max_loc = cv2.minMaxLoc(c)

                if max_val > CLEF_THRESH:
                    detected_clef = clef_num
                    detected_clef_dim = scaled_template.shape
                    detected_clef_loc = max_loc

                    # Write information on markup image
                    if detected_clef == 0:
                        clef_name = "Treble"
                    else:
                        clef_name = "Bass"

                    # Draw Rectangle around Clef
                    x, y = round(max_loc[0]), round(max_loc[1])
                    img_top = round(staff_top - Staff.vpadding * staff_height)
                    cv2.rectangle(markup_image, (x + staff.lines[0][0], y + img_top), (x + scaled_template.shape[1] + staff.lines[0][0], y + scaled_template.shape[0] + img_top),(0, 0, 255), 1)

                    # Write info next to clef
                    text_x = round(max_loc[0]+staff.lines[0][0]-scaled_template.shape[1]*1.8)
                    text_y = round(max_loc[1]+staff_top)
                    markup_image = cv2.putText(markup_image, clef_name, (text_x, text_y),
                                               cv2.FONT_HERSHEY_COMPLEX_SMALL, staff.gray_img.shape[1]/MARKUP_FONT_DIVIDER, (0, 0, 255))
                    break

                tune += tune_step

        assert detected_clef is not None, "Could not detect all clefs"

        # Set staff clef variables
        staff.clef = detected_clef
        staff.clef_dim = detected_clef_dim
        staff.clef_loc = detected_clef_loc

    return staffs, markup_image


# Finds the notes and their values in a staff. Sets the staff's notes variable and returns a marked up image
def detect_notes(staff, markup_image):
    staff_top = staff.lines[0][1]
    staff_height = staff.lines[-1][1] - staff_top

    # Get rid of staff lines
    kernel = np.ones((2, 1), np.uint8)
    staff_image = cv2.morphologyEx(staff.gray_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("s", staff_image)
    # cv2.waitKey(0)

    templates = [   # (Template Image, Counts)
        (cv2.imread(TEMPLATE_DIR+'/quarter-note.jpg'), 1),
        (cv2.imread(TEMPLATE_DIR+'/half-note.jpg'), 2),
        (cv2.imread(TEMPLATE_DIR+'/whole-note.jpg'), 4),
    ]

    notes = []  # ((x, y), counts)
    for template, counts in templates:
        # Create gray template
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Scale template to be the same height as the line spacing
        scale = (staff_height/4) / template.shape[0]
        scaled_template = cv2.resize(template, dsize=None, fx=scale, fy=scale)

        # Match template on to staff image
        c = cv2.matchTemplate(staff_image, scaled_template, cv2.TM_CCOEFF_NORMED)

        # Get all points at threshold
        _, thresh_img = cv2.threshold(c, thresh=NOTE_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        thresh_img = np.array(thresh_img, dtype=np.uint8)  # Make sure type is correct

        # Close note blobs
        kernel = np.ones(scaled_template.shape, np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("Thresh", thresh_img)
        # cv2.waitKey(0)

        # Find centroids of connected components
        _, _, _, centroids = cv2.connectedComponentsWithStats(thresh_img)

        for x, y in centroids[1::]:
            y += -1     # Small shift

            # Get note center
            cx = x + scaled_template.shape[1]/2
            cy = y + scaled_template.shape[0]/2 + int(staff_top - Staff.vpadding * staff_height)
            notes.append(((cx, cy), counts))

            letter, accidental, octave = staff.classify_note(cy)
            accidental_str = accidental if accidental is not None else ''

            # Draw Rectangle around note
            x, y = int(x), int(y)
            img_top = round(staff_top - Staff.vpadding * staff_height)
            cv2.rectangle(markup_image, (x+staff.lines[0][0], y+img_top), (x + scaled_template.shape[1]+staff.lines[0][0], y + scaled_template.shape[0]+img_top), (0, 0, 255), 1)

            # Write info text next to note
            text_coord1 = (round(x+scaled_template.shape[1]+staff.lines[0][0]), round(y-scaled_template.shape[0]*0.2 + img_top))
            text_coord2 = (round(x+scaled_template.shape[1]+staff.lines[0][0]), round(y+scaled_template.shape[0]*1.5 + img_top))

            markup_image = cv2.putText(markup_image, letter+accidental_str+str(octave), text_coord1, cv2.FONT_HERSHEY_COMPLEX_SMALL, staff.gray_img.shape[1]/MARKUP_FONT_DIVIDER, (0,0,255))
            markup_image = cv2.putText(markup_image, str(counts), text_coord2, cv2.FONT_HERSHEY_COMPLEX_SMALL, staff.gray_img.shape[1]/MARKUP_FONT_DIVIDER, (0,0,255))

    # cv2.imshow("Matches", match_image)
    # cv2.waitKey(0)

    notes = [((*staff.classify_note(coord[1]), counts), coord)
             for coord, counts in notes]   # Notes defined by (value, coordinates)
    staff.notes += notes
    staff.notes.sort(key=lambda note: note[1][0])     # Sort notes by x value (left to right)

    return markup_image


# Finds the rests and their values in a staff. Adds to the notes list of the staff and returns a marked up image
def detect_rests(staff, markup_image):
    tune_start, tune_stop, tune_step = 0.9, 1.1, 0.02     # Scale tuning variables

    staff_top = staff.lines[0][1]
    staff_height = staff.lines[-1][1] - staff_top

    templates = [  # (Template Image, Counts)
        (cv2.imread(TEMPLATE_DIR + '/quarter-rest.jpg'), 1),
        (cv2.imread(TEMPLATE_DIR + '/half-rest.jpg'), 2),
        (cv2.imread(TEMPLATE_DIR + '/whole-rest.jpg'), 4),
    ]

    rests = []
    for template, counts in templates:
        # Create grayscaled template
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Scale base
        scale_base = staff.gray_img.shape[0] / (2*template.shape[0])    # Makes template the same height as staff image

        # Find best scaling factor for template
        score_images = []    # Max score at each scale
        tune = tune_start
        while tune < tune_stop:
            scale = scale_base * tune

            scaled_temp = cv2.resize(template, dsize=None, fx=scale, fy=scale)  # Scale template

            c = cv2.matchTemplate(staff.gray_img, scaled_temp, cv2.TM_CCOEFF_NORMED)
            score_images.append((scaled_temp, c))
            tune += tune_step

        scaled_template, c = max(score_images, key=lambda x: np.amax(x[1]))     # Get score matrix with highest max

        # Get all points at threshold
        _, thresh_img = cv2.threshold(c, thresh=REST_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        thresh_img = np.array(thresh_img, dtype=np.uint8)  # Make sure type is correct

        # Close rest blobs
        temp_w = scaled_template.shape[1]
        kernel = np.ones((temp_w, temp_w), np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

        _, _, _, centroids = cv2.connectedComponentsWithStats(thresh_img)   # Get centroids of connected components

        for x, y in centroids[1::]:
            # Get note center
            cx = x + scaled_template.shape[1]/2
            cy = y + scaled_template.shape[0]/2 + int(staff_top - Staff.vpadding * staff_height)
            rests.append((counts, (cx, cy)))

            # Draw rectangle around rest
            x, y = int(x), int(y)
            img_top = round(staff_top - Staff.vpadding * staff_height)
            cv2.rectangle(markup_image, (x+staff.lines[0][0], y+img_top), (x + scaled_template.shape[1]+staff.lines[0][0], y + scaled_template.shape[0]+img_top), (0, 0, 255), 1)

            # Write information on markup image
            text_coord1 = (round(x + scaled_template.shape[1] + staff.lines[0][0]),
                                                             round(y + scaled_template.shape[0] * 0.4) + round(
                                                                 staff_top - Staff.vpadding * staff_height))
            text_coord2 = (round(x + scaled_template.shape[1] + staff.lines[0][0]),
                                                    round(y + scaled_template.shape[0] * 0.9) + round(
                                                        staff_top - Staff.vpadding * staff_height))
            markup_image = cv2.putText(markup_image, "Rest", text_coord1, cv2.FONT_HERSHEY_COMPLEX_SMALL, staff.gray_img.shape[1]/MARKUP_FONT_DIVIDER, (0,0,255))
            markup_image = cv2.putText(markup_image, str(counts), text_coord2, cv2.FONT_HERSHEY_COMPLEX_SMALL, staff.gray_img.shape[1]/MARKUP_FONT_DIVIDER, (0,0,255))

    staff.notes += rests
    staff.notes.sort(key=lambda note: note[1][0])   # Sort by x (left to right)

    return markup_image


# Finds the key signature accidentals in a staff. Adds them to the staff variable
def detect_signature_accidentals(staff, markup_image):
    tune_start, tune_stop, tune_step = 0.9, 1.1, 0.02  # Variables for scale tunes

    # Staff dimensions
    staff_top = staff.lines[0][1]
    staff_height = staff.lines[-1][1] - staff_top
    staff_line_spacing = staff_height/4

    templates = [
        (cv2.imread(TEMPLATE_DIR + '/sharp.jpg'), '#'),
        (cv2.imread(TEMPLATE_DIR + '/flat.jpg'), 'b'),
    ]

    # Get rid of staff lines
    kernel = np.ones((2, 1), np.uint8)
    staff_image = cv2.morphologyEx(staff.gray_img, cv2.MORPH_CLOSE, kernel)

    # Make subimage of just staff key signature
    signature_image = staff_image[:, staff.clef_loc[0]+staff.clef_dim[1]:staff.clef_loc[0]+2*staff.clef_dim[1]]

    # cv2.imshow("Key Signature", signature_image)
    # cv2.waitKey(0)

    for template, accidental in templates:
        # Create gray template
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Scale base
        scale_base = 2.4*staff_line_spacing / template.shape[0]    # Makes template the same height as staff image

        # Find best scaling factor for template
        scores_images = []    # Max score at each scale
        tune = tune_start
        while tune < tune_stop:
            scale = scale_base * tune

            scaled_temp = cv2.resize(template, dsize=None, fx=scale, fy=scale)  # Scale template

            c = cv2.matchTemplate(signature_image, scaled_temp, cv2.TM_CCOEFF_NORMED)
            scores_images.append((scaled_temp, c))
            tune += tune_step

        scaled_template, c = max(scores_images, key=lambda x: np.amax(x[1]))    # Get score matrix with highest max val

        # Get all points at threshold
        _, thresh_img = cv2.threshold(c, thresh=ACCIDENTAL_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        thresh_img = np.array(thresh_img, dtype=np.uint8)  # Make sure type is correct

        # cv2.imshow("Thresh", thresh_img)
        # cv2.waitKey(0)

        _, _, _, centroids = cv2.connectedComponentsWithStats(thresh_img)   # Get centroids of connected components

        for x, y in centroids[1::]:
            # Get accidentals location
            cx = x + scaled_template.shape[1]/2
            cy = y + scaled_template.shape[0]/2 + int(staff_top - Staff.vpadding * staff_height)

            # Find the note letter corresponding to that y value
            letter = staff.classify_note(cy)[0]

            if accidental == '#':
                staff.sharps.append(letter)  # Add letter to staffs sharps list
            elif accidental == 'b':
                staff.flats.append(letter)  # Add letter to staffs flats list

            # Draw rectangle around accidental
            x, y = round(x), round(y)
            img_top = round(staff_top - Staff.vpadding * staff_height)
            cv2.rectangle(markup_image, (x + staff.lines[0][0]+staff.clef_loc[0]+staff.clef_dim[1], y+img_top),
                          (x + staff.lines[0][0]+staff.clef_loc[0]+staff.clef_dim[1]+scaled_template.shape[1], y + img_top + scaled_template.shape[0]),
                          (0, 0, 255), 1)

            # Write accidental information on markup image
            text_coord = (round(cx+scaled_template.shape[1]+staff.lines[0][0]+signature_image.shape[1]), round(y-scaled_template.shape[0]*0.2 + staff_top - Staff.vpadding * staff_height))
            cv2.putText(markup_image, letter+accidental, text_coord, cv2.FONT_HERSHEY_COMPLEX_SMALL, staff.gray_img.shape[1]/MARKUP_FONT_DIVIDER, (0, 0, 255))

    return markup_image

