import cv2
import numpy as np
import math


def find_sheet_music(bgr_image):
    B_THRESH = 150

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Threshold saturation image
    _, thresh_image = cv2.threshold(gray_image, thresh=B_THRESH, maxval=255, type=cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image=bgr_image, contours=contours, color=(0, 0, 255), thickness=2, contourIdx=-1) #TESTING ONLY

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea)
    # Get largest contour
    contour = contours[-1]

    # Get the corner points of largest contour
    arclen = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * arclen, True)


    # Draw corner points TESTING ONLY
    # cv2.drawMarker(bgr_image, position=(approx[0][0][0], approx[0][0][1]), color=(255, 0, 0), #TL
    #                markerType=cv2.MARKER_CROSS, thickness=50)
    # cv2.drawMarker(bgr_image, position=(approx[1][0][0], approx[1][0][1]), color=(255, 0, 0), #BL
    #                markerType=cv2.MARKER_CROSS, thickness=50)
    # cv2.drawMarker(bgr_image, position=(approx[2][0][0], approx[2][0][1]), color=(255, 0, 0), #BR
    #                markerType=cv2.MARKER_CROSS, thickness=50)
    # cv2.drawMarker(bgr_image, position=(approx[3][0][0], approx[3][0][1]), color=(255, 0, 0), #TR
    #                markerType=cv2.MARKER_CROSS, thickness=50)

    # Order corner points based on x-coord
    corner_points = []
    for i in range(4):
        for j in range(len(corner_points)):
            if(approx[i][0][0] < corner_points[j][0]):
                temp = corner_points[j]
                corner_points[j] = approx[i][0]
                corner_points.append(temp)
                break
        else:
            corner_points.append(approx[i][0])

    # Get new image dimensions based on corner points
    x = abs(corner_points[1][0] - corner_points[0][0])
    y = abs(corner_points[1][1] - corner_points[0][1])
    sheet_width = int(math.sqrt(pow(x, 2) + pow(y, 2)))

    x = abs(corner_points[2][0] - corner_points[0][0])
    y = abs(corner_points[2][1] - corner_points[0][1])
    sheet_height = int(math.sqrt(pow(x, 2) + pow(y, 2)))

    # Find homography
    pts1 = np.array([corner_points[0], corner_points[1], corner_points[2], corner_points[3]]) # Left to right on image (vary based on ration)
    if (corner_points[1][1] < corner_points[0][1]):
        pts2 = np.array([[0, sheet_height], [0, 0], [sheet_width, sheet_height], [sheet_width, 0]]) # BL TL BR TR
    else:
        pts2 = np.array([[0, 0], [0, sheet_height], [sheet_width, 0], [sheet_width, sheet_height]]) # TL BL TR BR
    H1, _ = cv2.findHomography(srcPoints=pts1, dstPoints=pts2)

    # Warp and get sheet music image
    bgr_ortho = cv2.warpPerspective(bgr_image, H1, (sheet_width, sheet_height))
    return bgr_ortho

    # Print result TESTING ONLY
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", bgr_ortho)
    # cv2.waitKey(0)