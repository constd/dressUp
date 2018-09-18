import argparse
import pickle
import youtube_dl
from .resnet import resnet_filter
import tensorflow as tf
import requests
import re


def get_ids(opts):
    vid_ids = []
    labels = []

    labels_we_are_looking_for = set(opts.desired_labels)
    files = tf.gfile.Glob(opts.tfrecordspath)
    for video_lvl_record in files:
        for example in tf.python_io.tf_record_iterator(video_lvl_record):
            tf_example = tf.train.Example.FromString(example)
            current_vid_id = tf_example.features.feature["id"].bytes_list.value[0].decode(encoding="UTF-8")
            current_labels = tf_example.features.feature["labels"].int64_list.value
            if labels_we_are_looking_for.intersection(current_labels):
                vid_ids.append(current_vid_id)
                labels.append(current_labels)

    youtube_dataset_video_ids = []
    reg = re.compile(r'i\("(?P<dataset_id>\S*)","(?P<youtube_id>\S*)"\);')

    for vid_id in vid_ids:
        rqst = requests.get("http://data.yt8m.org/2/j/i/{}/{}.js".format(vid_id[:2], vid_id))
        if not rqst:
            continue
        r = reg.match(rqst.text)
        if not r:
            continue
        youtube_dataset_video_ids.append([r.groupdict()["dataset_id"], r.groupdict()["youtube_id"]])

    with open(opts.outfile, "rb") as f:
        pickle.dump(youtube_dataset_video_ids, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve Ids for youtube videos based on a list of desired labels")
    parser.add_argument("--desired_labels", nargs="+", type=int,
                        default=[71, 849, 1313])
    parser.add_argument("--outfile", help="the file where all the ids are finally stored",
                        default="yt_ids.pkl")
    parser.add_argument("--tfrecordspath", help="(glob-able) path where all our tfrecord files are stored", 
                        default="~/data/yt8m/**/*.tfrecord")
    opts = parser.parse_args()
    get_ids(opts)
