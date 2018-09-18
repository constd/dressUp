"""
This script loads the list of ids
"""
import pickle
import argparse
import logging
import sys
import streamlink
import os.path
import numpy as np
from dressup.resnet import resnet_filter_img
try:
    import cv2
except ImportError:
    sys.stderr.write("This example requires opencv-python is installed")
    raise


log = logging.getLogger(__name__)
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


def stream_to_url(url, quality='best'):
    streams = streamlink.streams(url)
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError("No steams were available")


def detect_faces(cascade, frame, scale_factor=1.25, min_neighbors=6):
    frame_copy = frame.copy()
    frame_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, w, h) in faces:
        fx, fy, fxw, fyh = max(0, x - int(1.5*w)), max(0, y - int(.6*h)), min(frame_copy.shape[1], x + int(2.25*w)), min(frame_copy.shape[0], y + 9*h)
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), GREEN, 1)
        cv2.rectangle(frame_copy, (fx, fy), (fxw, fyh), RED, 1)
        text = resnet_filter_img(resizeAndPad(frame_copy[fy:fyh, fx:fxw], (224, 224)), set(['gown', 'mosquito_net', 'hoopskirt']))
        cv2.putText(frame_copy, text, (fx, fy), FONT, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_copy


def main(url, quality='best', fps=30.0):
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, 'haarcascade_frontalface_default.xml'))
    stream_url = stream_to_url(url, quality)
    log.info("Loading stream {0}".format(stream_url))
    cap = cv2.VideoCapture(stream_url)
    ret, frame = cap.read()
    print(frame.shape)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = None
    frame_time = int((1.0 / fps) * 1000.0)
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                frame_f = detect_faces(face_cascade, frame, scale_factor=1.2)
                if out is None:
                    out = cv2.VideoWriter('outpy.avi', fourcc, 30, (frame_f.shape[0], frame_f.shape[1]))
                else:
                    out.write(frame_f)
                cv2.imshow('frame', frame_f)
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break
            else:
                break
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Face detection on streams via Streamlink")
    parser.add_argument("ytidpath", help="Path to file that contains the youtube ids",)
    parser.add_argument("urlindex", help="Stream to play", default=5, type=int)
    parser.add_argument("--stream-quality", help="Requested stream quality [default=best]",
                        default="best", dest="quality")
    parser.add_argument("--fps", help="Play back FPS for opencv [default=30]",
                        default=30.0, type=float)
    opts = parser.parse_args()
    with open(opts.ytidpath, 'rb') as f:
        yt_ids = pickle.load(f)

    main('https://www.youtube.com/watch?v={}'.format(yt_ids[opts.urlindex][1]), opts.quality, opts.fps)
