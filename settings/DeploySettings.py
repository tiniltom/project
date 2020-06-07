import numpy as np
import cv2
PATH_TO_MODEL_CHECKPOINTS = "D:/UPWORK/ViolenceDetection-master/save_model/save_epoch_12/ViolenceNet.ckpt"


CHANGE_JUDGEMENT_THRESHOLD = 3

DISPLAY_IMAGE_SIZE = 500

BORDER_SIZE = 5
FIGHT_BORDER_COLOR = (0, 0, 255)
NO_FIGHT_BORDER_COLOR = (0, 255, 0)


font = cv2.FONT_HERSHEY_SIMPLEX


org = (50, 50)


fontScale = 1


color1 = (0, 255, 0)
color2 = (0, 0, 255)

thickness = 2

