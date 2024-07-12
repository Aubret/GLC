import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='/home/fias/postdoc/datasets/ego4d', type=str)
args = parser.parse_args()


names = {}
with open("data/full_ego4d_gaze.csv", "r") as f:
    datareader = csv.reader(f)
    for r in datareader:
        name = r[0].split("/")[0 ] +".mp4"
        if name in ['4e07da0c-450f-4c37-95e9-e793cb5d8f7f.mp4',
                    '5819e52c-4e12-4f86-ad69-76fc215dfbcb.mp4',
                    '83081c5a-8456-44d8-af67-280034f8f0a6.mp4',
                    'a77682da-cae7-4e68-8580-6cb47658b23f.mp4']:
            continue
        names[name] = 1

# f = open(f"data/filelist0.csv", "w")
# for n in names.keys():
#     f.write(n+"\n")


source_path = f'{args.src}/full_scale.gaze'

idx = 1
video_list = []
for p in sorted(os.listdir(source_path)):
    if not os.path.splitext(p)[-1] == '.mp4':
        continue
    if p in names:
        continue
    video_list.append(p[:-4])
    if len(video_list) >= 100:
        f = open(f"data/filelist{idx}.csv", "w")
        for n in video_list:
            f.write(n+"\n")
        idx += 1
        video_list = []
f = open(f"data/filelist{idx}.csv", "w")
for n in video_list:
    f.write(n+"\n")


