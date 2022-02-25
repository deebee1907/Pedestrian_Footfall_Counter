'''
    _____________________________________
    PEDESTRIAN FOOTFALL DETECTION PROJECT
    -------------------------------------
    Group Members:

    > Aditi Sinha
      CS-3A
      Roll No.-04
    > Doel Bhattacharya
      CS-3A
      Roll No.-24
    > Abhinaba Chakraborty
      CS-3H
      Roll No.-01
    > Swati Ghosh
      CS-3H
      Roll No.-73
    --------------------------------------
'''


# Importing required modules.
import cv2
import imutils


# Function to detect and count number of pedestrians in a given video footage.
def DrawDetected(image):
    coordinates, weights = HOGCV.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)

    detected = 1
    for x, y, w, h in coordinates:
        # Draws a box around a detected pedestrian in the video footage.
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'person {detected}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        detected += 1
        # Sending out a warning for exceeded limit of number of pedestrians.
        if detected >= 6:
            cv2.putText(image, 'PEOPLE LIMIT EXCEEDED!!!', (270, 270), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

    # Displaying number of pedestrians detected.
    cv2.putText(image, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(image, f'Total Persons : {detected - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('Video Footage', image)

    return image


# Function to iteratively detect pedestrians for each frame in a given video sample.
def DetectVideo(path, writer):
    getVideo = cv2.VideoCapture(path)
    flag, image = getVideo.read()
    if flag is False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while getVideo.isOpened():
        # Checking to see if video has been successfully read.
        flag, image = getVideo.read()

        if flag:
            image = imutils.resize(image, width=min(800, image.shape[1]))
            image = DrawDetected(image)

            if writer is not None:
                writer.write(image)

            # Entering the key 'q' on the keyboard exits from the pedestrian detection footage.
            keyInput = cv2.waitKey(1)
            if keyInput == ord('q'):
                print('Closing the Video...')
                break
        else:
            break
    getVideo.release()
    cv2.destroyAllWindows()


# Function to initialize video footage and convert the video format to a suitable format.
def PedestrianDetector():
    videoName = 'walking.avi'

    writer = cv2.VideoWriter(None, cv2.VideoWriter_fourcc(*'DIVX'), 10, (600, 600))

    print('[INFO] Opening Video from path.')
    DetectVideo(videoName, writer)


# Main function.
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
PedestrianDetector()
