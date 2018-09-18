import sys
import pickle
from multiprocessing import Pool


def main(list_of_videos):
    import youtube_dl
    class MyLogger(object):
        def debug(self, msg):
            pass
        def warning(self, msg):
            pass
        def error(self, msg):
            print(msg)

    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    ydl_opts = {
    'outtmpl': '/Volumes/bilongo/yt_videos/%(id)s.%(ext)s',
    'format': 'bestvideo/best',
    'noplaylist' : True,
    'keepvideo': True,
    'geo_bypass': True,
    'writeinfojson': True,
    'logger': MyLogger(),
    'progress_hooks': [my_hook],
    }
    for vid in list_of_videos:
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([vid[1]])
        except Exception as e:
            print(e)

if __name__ == '__main__':
    # sys.argv[0] is the path to the file that contains a list of lists with the youtube ids we want to download
    with open(sys.argv[0], 'rb') as f:
        yt_ids = pickle.load(f)
    # split the list of ids in sys.argv[1] equal parts
    n = len(yt_ids)//int(sys.argv[2])
    yts = [yt_ids[i:i + n] for i in range(0, len(yt_ids), n)]
    # download the sys.argv[2] chunk of the list
    main(yts[int(sys.argv[3])])
