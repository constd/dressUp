from bs4 import BeautifulSoup as bs
import boltons.fileutils
import json
import logging
import numpy as np
import os
from pathlib import Path
import requests
import time

logger = logging.getLogger(__name__)


def fetch_board_rss(board_rss_link, cache_dir=None):
    board_name = Path(board_rss_link).stem
    cache_file = ((Path(cache_dir) / f"{board_name}.json")
                  if cache_dir else None)
    if cache_file is not None and cache_file.exists():
        with open(cache_file, 'r') as fh:
            all_pins = json.load(fh)

    else:
        r = requests.get(board_rss_link)
        xml = bs(r.text, "xml")

        all_pins = [item.link.contents[0] for item in xml.findAll('item')]

        if cache_file is not None:
            boltons.fileutils.mkdir_p(cache_dir)
            with open(cache_file, 'w') as fh:
                json.dump(all_pins, fh)
    return all_pins


def fetch_pin(pin_link):
    r = requests.get(pin_link)
    html = bs(r.text)

    return html


def findAllByKey(key, dictionary):
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in findAllByKey(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    for result in findAllByKey(key, d):
                        yield result


def dl_images_from_pin(pin_link, save_images=True, download_dir="./images",
                       which_image='orig', metadata_file="images.json"):
    """Downloads all the images linked to a given pin url.

    Parameters
    ----------
    pin_link : str
        URL of a pintrest pin

    save_images : bool
        Try to save the images or not.
    """
    r = requests.get(pin_link)
    html = bs(r.content, 'lxml')

    data = json.loads(next(html.find(id="initial-state").children))
    images = list(findAllByKey("images", data))

    metadata_path = Path(metadata_file)
    if metadata_path.exists():
        with open(metadata_path, 'r') as fh:
            metadata = json.load(fh)
    else:
        metadata = {}

    for img_size_dict in images:
        img_url = img_size_dict[which_image]['url']
        metadata[os.path.basename(img_url)] = img_url

    if save_images:
        boltons.fileutils.mkdir_p(download_dir)

        for img_size_dict in images:
            img_url = img_size_dict[which_image]['url']
            basename = os.path.basename(img_url)

            r = requests.get(img_url)
            write_path = Path(download_dir) / basename
            with open(write_path, 'wb') as fh:
                fh.write(r.content)

    # Write the new metadata
    with open(metadata_path, 'w') as fh:
        json.dump(metadata, fh, indent=2)

    return images


def download_all_pins_at_random_interval(
        pin_list, time_between_bursts=60, burst_count=8):
    pin_list_index = 0
    counter = 0
    while pin_list_index < len(pin_list):
        for pin_link in pin_list:
            print(f"Downloading from {pin_link}")
            dl_images_from_pin(pin_link)

            counter += 1
            if counter >= burst_count:
                wait_time = (np.random.poisson(time_between_bursts) +
                             np.random.random())
                print(f"Burst complete - Sleeping for {wait_time}s")
                time.sleep(wait_time)
                counter = 0

            else:
                # Sleep some small amount
                sleep_time = np.random.random()
                print(f"sleeping for {sleep_time}s "
                      f"(count={counter} / {burst_count})")
                time.sleep(sleep_time)
