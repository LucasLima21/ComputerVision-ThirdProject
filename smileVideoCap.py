import cv2
import dlib


def faceLandmarks(im):

    # Path for the detection model, you can download it from here: https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    
    # Create object to detect the face
    faceDetector = dlib.get_frontal_face_detector()

    # Create object to detect the facial landmarks
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    # Detect faces
    faceRects = faceDetector(im, 0)

    # Initialize landmarksAll array
    landmarksAll = []

    # For each face detected in the image, this chunk of code creates a ROI around the face and pass it as an argument to the 
    # facial landmark detector and append the result to the array landmarks 
    for i in range(0, len(faceRects)):
        newRect = dlib.rectangle(int(faceRects[i].left()),
                            int(faceRects[i].top()),
                            int(faceRects[i].right()),
                            int(faceRects[i].bottom()))
        landmarks = landmarkDetector(im, newRect)
        landmarksAll.append(landmarks)

    return landmarksAll, faceRects


def renderFacialLandmarks(im, landmarks):
    
    # Convert landmarks into iteratable array
    points = []
    [points.append((p.x, p.y)) for p in landmarks.parts()]

    # Loop through array and draw a circle for each landmark
    for p in points:
        cv2.circle(im, (int(p[0]),int(p[1])), 2, (255,0,0),-1)

    # Return image with facial landmarks 
    return im

# Read an image to a variable
# im = cv2.imread("girlSmile.png")
im = cv2.VideoCapture(0)
# Get landmarks using the function created above
while True:
    _, image = im.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks, _ = faceLandmarks(gray)
    # Render the landmarks on the first face detected. You can specify the face by passing the desired index to the landmarks array.
    # In this case, one face was detected, so I'm passing landmarks[0] as the argument.
    faceWithLandmarks = renderFacialLandmarks(image, landmarks[0])


    # Calculate lips width
    lips_width = abs(landmarks[0].parts()[49].x - landmarks[0].parts()[55].x)

    # Calculate jaw width
    jaw_width = abs(landmarks[0].parts()[3].x - landmarks[0].parts()[15].x)

    # Calculate the ratio of lips and jaw widths
    ratio = lips_width/jaw_width

    # Evaluate ratio
    if ratio > 0.32 :
        result = "Smile"
    else:
        result = "No Smile"

    # Add result text to the image that will be displayed
    cv2.putText(faceWithLandmarks, result, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
im.release()



# Display image
# cv2.imshow("Smile Detection", faceWithLandmarks)
# cv2.waitKey(0)
