import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='/home/fias/postdoc/datasets/ego4d', type=str)
parser.add_argument('--select', default=1, type=int)
args = parser.parse_args()

lines = [item[0] for item in csv.reader(open(f"data/filelist{str(args.select)}.csv", "r"))]
writer = csv.writer(open(f"data/fullfilelist{str(args.select)}.csv", "w"))
# writer = csv.writer(f"data/fullfilelist{str(args.select)}.csv")

full_list = []
for l in lines:
    path = os.path.join(args.src, "clips.gaze", l)
    for p in os.listdir(path):
        if p[-4:] == ".mp4":
            writer.writerow([os.path.join(l,p)])



