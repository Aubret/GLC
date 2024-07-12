# Getting Started with PySlowFast

This document provides explinas how to generate ego4d data

# Preprocess

### Download Ego4d videos

```
ego4d --output_directory=SRC --yes --datasets video_540ss
ego4d --output_directory=SRC --yes --datasets gaze
```

### Split videos in 94 sets of videos without gaze annotations

```
python3 data/split.py --src {SRC}
```

### Create small clips from big videos and annotations

```
for i in {0..94}
do
    python3 data/preprocessing.py --src {SRC}/v2/ --dest {DEST}$i --select data/filelist$i.csv
    python3 data/record_splits.py --src {DEST}$i --select $i
done
```

### Data generation

First, videos with true labels
```
i=0
python3 tools/run_net.py --cfg ./configs/Ego4d/MVIT_B_16x4_CONV.yaml 
    TRAIN.CHECKPOINT_FILE_PATH /checkpoints/MViT_Ego4D_ckpt.pyth
    GENERATE.ENABLE True
    GENERATE.GENERATE_FILE data/fullfilelist 
    GENERATE.PATH_DATASET {DESTHDF5}
    GENERATE.APPEND $i
    GENERATE.LOG False 
    GENERATE.TRUE_LABEL True 
    DATA.PATH_PREFIX {DEST}/clips.gaze 
    NUM_GPUS {NGPU}
```

Then videos with generated labels

```
for i in {1..94}
do
    python3 tools/run_net.py --cfg ./configs/Ego4d/MVIT_B_16x4_CONV.yaml 
        TRAIN.CHECKPOINT_FILE_PATH /checkpoints/MViT_Ego4D_ckpt.pyth
        GENERATE.ENABLE True
        GENERATE.GENERATE_FILE data/fullfilelist 
        GENERATE.PATH_DATASET {DESTHDF5}
        GENERATE.APPEND $i
        GENERATE.LOG False 
        GENERATE.TRUE_LABEL False 
        DATA.PATH_PREFIX {DEST}/clips.gaze 
        NUM_GPUS {NGPU}
done
```
