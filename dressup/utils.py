import numpy as np
import cv2
from dressup.resnet import resnet_filter_img

GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def get_person_bounding_box(x, y, w, h):
    return max(0, x - int(1.5*w)), max(0, y - int(.6*h)), min(frame_copy.shape[1], x + int(2.25*w)), min(frame_copy.shape[0], y + 9*h)

def detect_faces(cascade, frame, scale_factor=1.25, min_neighbors=6):
    frame_copy = frame.copy()
    frame_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, w, h) in faces:
        fx, fy, fxw, fyh = get_person_bounding_box(x, y, w, h)
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), GREEN, 1)  # generate face bounding box
        cv2.rectangle(frame_copy, (fx, fy), (fxw, fyh), RED, 1)  # generate person bounding box
        text = resnet_filter_img(resizeAndPad(frame_copy[fy:fyh, fx:fxw], (224, 224)), set(['gown', 'mosquito_net', 'hoopskirt']))
        cv2.putText(frame_copy, text, (fx, fy), FONT, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_copy