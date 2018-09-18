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
import cv2
from dressup.utils import detect_faces, resizeAndPad, RED, GREEN, FONT


log = logging.getLogger(__name__)


def stream_to_url(url, quality='best'):
    streams = streamlink.streams(url)
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError("No steams were available")


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
