# Multiscale-custom-tag-matching

Designed for Cognitive Robotics (15-494/694) at CMU. Presentation: https://github.com/SlashDK/Multiscale-custom-tag-matching/blob/master/Custom%20Marker%20Recognizer%20Final%20Project.pptx 

Program to take in a template line / handdrawn image, and locate similar ones in a video stream. Can be used on its own, but it also designed to be used with the Anki Cozmo on top of https://github.com/touretzkyds/cozmo-tools. Refer to https://github.com/touretzkyds/cozmopedia/wiki for more information.

To run:
1) For a live webcam stream (cv2.VideoCapture), run match.py and call match(template, confidenceThresh, overlapThresh), where
2) For comparing to a set of images in a folder, run matchWithFiles.py and call match(template, imagefolder=None, confidenceThresh=0.2, overlapThresh=0.2, showEdges=False, showEdges2=False).

Parameters:
template should be a string to the location of template image. Code uses cv2.imread(template) to read image.

imagefolder should be a string to the location of template image. Code uses glob.glob(imagefolder + "\\*.jpg") to read images. Only jpg images tested currently.

confidenceThresh should be between 0.0 and 1.0, with a default setting of 0.2. The higher the value, the closer the detected images must be to the template. Affected by image placement, lighting conditions etc., so lower values are preferred unless running over files. Recommended values 0.1-0.35 for hand drawings. Upto 0.6 for printouts of same marker. Can be even higher for files.

overlapThresh should be between 0.0 and 1.0, with a default setting of 0.2. Percentage over which bounding boxes should be combined. Ideally should be very low (<0.2), but can be increased if markers very densely packed (if markers cannot have non-intersecting bounding boxes). 

showEdges visualizes the running of the algorithm

showEdges2 visualized all bounding boxes without merging.
