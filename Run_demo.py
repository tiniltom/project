import os
import sys
import cv2
import numpy as np
import time
from src.ViolenceDetector import *
import settings.DeploySettings as deploySettings
import settings.DataSettings as dataSettings
import src.data.ImageUtils as ImageUtils
from tkinter import filedialog


class VideoSavor:
    def AppendFrame(self, image_):
        self.outputStream.write(image_)

    def __init__(self, targetFileName, videoCapture):
        width = int(deploySettings.DISPLAY_IMAGE_SIZE)
        height = int(deploySettings.DISPLAY_IMAGE_SIZE)
        frameRate = int(videoCapture.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        self.outputStream = cv2.VideoWriter(targetFileName + ".avi", codec, frameRate, (width, height))


def PrintUnsmoothedResults(unsmoothedResults_):
    print("Unsmoothed results:")
    print("\t [ ")
    print("\t   ", end='')
    for i, eachResult in enumerate(unsmoothedResults_):
        if i % 10 == 9:
            print(str(eachResult) + ", ")
            print("\t   ", end='')

        else:
            print(str(eachResult) + ", ", end='')

    print("\n\t ]")


def DetectViolence(PATH_FILE_NAME_OF_SOURCE_VIDEO, saveresult):
    violenceDetector = ViolenceDetector()
    videoReader = cv2.VideoCapture(PATH_FILE_NAME_OF_SOURCE_VIDEO)
    if saveresult == True:
        videoSavor = VideoSavor(PATH_FILE_NAME_OF_SOURCE_VIDEO + "_Result", videoReader)

    listOfForwardTime = []
    isCurrentFrameValid, currentImage = videoReader.read()
    while isCurrentFrameValid:
        netInput = ImageUtils.ConvertImageFrom_CV_to_NetInput(currentImage)

        startDetectTime = time.time()
        isFighting = violenceDetector.Detect(netInput)
        endDetectTime = time.time()
        listOfForwardTime.append(endDetectTime - startDetectTime)

        targetSize = deploySettings.DISPLAY_IMAGE_SIZE - 2 * deploySettings.BORDER_SIZE
        currentImage = cv2.resize(currentImage, (targetSize, targetSize))
        if isFighting:
            resultImage = cv2.copyMakeBorder(currentImage,
                                             deploySettings.BORDER_SIZE,
                                             deploySettings.BORDER_SIZE,
                                             deploySettings.BORDER_SIZE,
                                             deploySettings.BORDER_SIZE,
                                             cv2.BORDER_CONSTANT,
                                             value=deploySettings.FIGHT_BORDER_COLOR)
            # Using cv2.putText() method
            resultImage = cv2.putText(resultImage, 'Violence :(', deploySettings.org, deploySettings.font,
                                      deploySettings.fontScale, deploySettings.color2, deploySettings.thickness,
                                      cv2.LINE_AA)
        else:
            resultImage = cv2.copyMakeBorder(currentImage,
                                             deploySettings.BORDER_SIZE,
                                             deploySettings.BORDER_SIZE,
                                             deploySettings.BORDER_SIZE,
                                             deploySettings.BORDER_SIZE,
                                             cv2.BORDER_CONSTANT,
                                             value=deploySettings.NO_FIGHT_BORDER_COLOR)
            resultImage = cv2.putText(resultImage, 'NonViolence :)', deploySettings.org, deploySettings.font,
                                      deploySettings.fontScale, deploySettings.color1, deploySettings.thickness,
                                      cv2.LINE_AA)

        cv2.imshow("Violence Detection", resultImage)
        if saveresult == True: 
            videoSavor.AppendFrame(resultImage)

        userResponse = cv2.waitKey(1)
        if userResponse == ord('q'):
            videoReader.release()
            cv2.destroyAllWindows()
            break

        else:
            isCurrentFrameValid, currentImage = videoReader.read()

    PrintUnsmoothedResults(violenceDetector.unsmoothedResults)
    averagedForwardTime = np.mean(listOfForwardTime)
    print("Averaged Forward Time: ", averagedForwardTime)


if __name__ == '__main__':
    saveresult = False
    PATH_FILE_NAME_OF_SOURCE_VIDEO = filedialog.askopenfilename()
    DetectViolence(PATH_FILE_NAME_OF_SOURCE_VIDEO, saveresult)
