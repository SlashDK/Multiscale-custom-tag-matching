# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
np.set_printoptions(threshold=np.nan)

# Function to combine intersecting bounding boxes


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
#
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def match(template, imagefolder=None, confidenceThresh=0.2, overlapThresh=0.2, showEdges=False, showEdges2=False):

    # load the image image, convert it to grayscale, and detect edges
    numScales = 20
    threshold = confidenceThresh
    template = cv2.imread(template)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 100, 200)
    template = cv2.resize(template, (50, 50))
    (h, w) = template.shape[:2]
    cv2.imwrite("template.jpg", template)
    cv2.imshow("Template", template)
    # cam = cv2.VideoCapture(0)

    cv2.namedWindow('Image')

    # If running over files instead of video, comment out cv2.VideoCapture
    # and while(True) and uncomment the two lines below
    for imagePath in glob.glob(imagefolder + "\\*.jpg"):
        image = cv2.imread(imagePath)
    # while (True):
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region

        # ret_val, image = cam.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        # loop over the scales of the image
        loc = []
        crec = []
        cv2.imshow("Image", image)
        for scale in np.linspace(0.2, 1.0, numScales)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # print(resized.shape)
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < h or resized.shape[1] < w:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 100, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)

            loc = np.where(result >= threshold)
            for pt in zip(*loc[::-1]):
                startend = (int(pt[0] * r), int(pt[1] * r),
                            int((pt[0] + w) * r), int((pt[1] + h) * r))
                crec.append(startend)

            # check to see if the iteration should be visualized
                if (showEdges):
                    # draw a bounding box around the detected region
                    clone = np.dstack([edged, edged, edged])
                    cv2.rectangle(clone, (startend[0], startend[1]), (startend[
                                  2], startend[3]), (0, 0, 255), 2)
                    cv2.imshow("Visualize", clone)
                    cv2.waitKey(1)

        boxes = non_max_suppression_fast(np.array(crec), overlapThresh)
        for box in boxes:
            foundIn = None
            cv2.rectangle(image, (box[0], box[1]),
                          (box[2], box[3]), (255, 255, 255), 1)
            croppedGray = gray[box[1]:box[3], box[0]:box[2]]
            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                # resize the image according to the scale, and keep track
                # of the ratio of the resizing

                resized = imutils.resize(
                    croppedGray, width=int(croppedGray.shape[1] * scale))
                r = croppedGray.shape[1] / float(resized.shape[1])

                # if the resized image is smaller than the template, then break
                # from the loop
                if resized.shape[0] < h or resized.shape[1] < w:
                    break

                # detect edges in the resized, grayscale image and apply template
                # matching to find the template in the image
                edged = cv2.Canny(resized, 100, 200)
                resultIn = cv2.matchTemplate(
                    edged, template, cv2.TM_CCORR_NORMED)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resultIn)
                # if we have found a new maximum correlation value, then update
                # the bookkeeping variable
                if (showEdges2):
                    # draw a bounding box around the detected region
                    clone = np.dstack([edged, edged, edged])
                    cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                                  (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
                    cv2.imshow("Visualize", clone)
                    cv2.waitKey(10)
                if foundIn is None or maxVal > foundIn[0]:
                    foundIn = (maxVal, maxLoc, r)

            (maxVal, maxLoc, r) = foundIn
            maxLoc = (maxLoc[0] + box[0] / r,  maxLoc[1] + box[1] / r)

            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))

            # draw a bounding box around the detected result and display the
            # image
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 0, 255), 2)

        cv2.imshow("Image", image)
        cv2.imshow("Edges", cv2.Canny(gray, 100, 200))
        cv2.waitKey(0)
        # if (cv2.waitKey(1) == 27):
        #     break  # esc to quit

match("5050.jpg", "Trained", 0.22, 0.05, False, True)
