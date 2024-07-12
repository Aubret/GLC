#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""
import csv
import io
import sys

import h5py
import numpy as np
import os
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.metrics as metrics
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader, transform
from slowfast.datasets.utils import tensor_unnormalize
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, TestGazeMeter
from slowfast.utils.utils import frame_softmax
from generate_gaze import show_cam_on_image

logger = logging.get_logger(__name__)



def crop_gaze_image(image, gaze, gaze_size, max = (64, 64)):
    scale = min(image.shape[1], image.shape[2])/min(*max)
    gaze[0] = scale*(gaze[0]- max[0]/2) + image.shape[2]/2
    gaze[1] = scale*(gaze[1]- max[1]/2) + image.shape[1]/2
    # if gaze_size == min(image.shape[1], image.shape[2]):
    #     return image, gaze[0:2]

    # print(gaze, cpt, image.shape)
    gaze_size = gaze_size // 2
    i,j,h,b = (gaze[0].item() - gaze_size, gaze[0].item() + gaze_size,
               gaze[1].item() - gaze_size, gaze[1].item() + gaze_size)
    xy_shift = 0
    hb_shift = 0
    # print(i,j,h,b)
    if i < 0:
        xy_shift = -i + 1
    if j >= image.shape[2]:
        xy_shift = image.shape[2] - j

    if h < 0:
        hb_shift = -h + 1

    if b >= image.shape[1]:
        hb_shift = image.shape[1] - b


    i += xy_shift
    j += xy_shift
    h += hb_shift
    b += hb_shift
    gaze[0:1] += xy_shift
    gaze[1:2] += hb_shift
    return image[:, round(h):round(b), round(i):round(j)], gaze[0:2]


def normalize_map(preds):
    preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
    preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (
                preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
    preds_rescale = preds_rescale.view(preds.size())
    return preds_rescale

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestGazeMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    csv_file = open(f"{cfg.GENERATE.PATH_DATASET}/dataset{cfg.GENERATE.APPEND}.csv", "w")
    csv_writer = csv.writer(csv_file)

    # object_dataset = category_dataset
    partition = cfg.GENERATE.APPEND
    print(test_loader.dataset.num_videos*(cfg.DATA.NUM_FRAMES - cfg.DATA.SAMPLING_RATE))
    with h5py.File(f'{cfg.GENERATE.PATH_DATASET}data{partition}.hdf5', 'w') as hf:
        # frame_number = test_loader.dataset.num_videos*(cfg.DATA.NUM_FRAMES - cfg.DATA.SAMPLING_RATE)
        frames = []
        dataset_number = 0
        # if not cfg.GENERATE.APPEND:
        for _ in cfg.GENERATE.GAZE_SIZE:
            frames.append([])
        cpt = 0
        h5_index = 0
        for cur_iter, (inputs, labels, labels_hm, video_idx, meta, orig_inputs) in enumerate(test_loader):
            # if cpt < cfg.GENERATE.APPEND:
            #     cpt += 48
            #     continue
            batch_size = inputs[0].shape[0]
            inputs = [torch.cat(inputs[0].split(8, dim=2)[:-1], dim=0)]
            new_batch_size = inputs[0].shape[0]
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    inputsc = [0]*len(inputs)
                    for i in range(len(inputs)):
                        inputsc[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputsc = inputs.cuda(non_blocking=True)
                preds = model(inputsc)
            else:
                preds = model(inputs)

            test_meter.data_toc()
            orig_inputs = [torch.cat(orig_inputs[0].split(8, dim=2)[:-1], dim=0)]
            labels = torch.cat(labels.split(8, dim=1)[:-1], dim=0)
            labels_hm = torch.cat(labels_hm.split(8, dim=1)[:-1], dim=0)


            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, labels_hm, video_idx = du.all_gather([preds, labels, labels_hm, video_idx])

            # PyTorch
            if cfg.NUM_GPUS:  # compute on cpu
                preds, labels, labels_hm, video_idx = preds.cpu(),labels.cpu(),labels_hm.cpu(),video_idx.cpu()


            preds = frame_softmax(preds, temperature=2)  # KLDiv
            preds_rescale = normalize_map(preds)

            f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels,dataset=cfg.TEST.DATASET)
            auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET)
            test_meter.iter_toc()
            test_meter.update_stats(f1, recall, precision, auc, preds=preds_rescale, labels_hm=labels_hm,labels=labels)
            test_meter.log_iter_stats(cur_iter)

            # We iterate and generate all the data
            assert len(inputs) == 1
            for i in range(inputs[0].size(0)):

                data_index = int(i*batch_size/new_batch_size)
                video_split = meta["path"][data_index].split("/")
                index_video = video_idx[data_index]
                video_name = video_split[-2]
                timestamp_split = video_split[-1].split("_")
                ts1 = timestamp_split[-2][1:]
                ts2 = timestamp_split[-1][1:-3]

                framest = inputs[0][i]
                original_framest = orig_inputs[0][i]

                for k in range(framest.shape[1]):
                    global_index = meta["index"][data_index,k]
                    original_frame = original_framest[:,k]
                    orig_frame = (original_frame*cfg.DATA.STD[0])+cfg.DATA.MEAN[0]

                    # get gaze location
                    if cfg.GENERATE.TRUE_LABEL:
                        lab = labels[i,k]*64
                    else:
                        predsik_max = torch.argmax(preds_rescale[i, 0, k].view(-1))
                        predsiki, predsikj = predsik_max % preds_rescale.shape[3], predsik_max // preds_rescale.shape[3]
                        lab = torch.cat((predsiki.view(1), predsikj.view(1)), dim=0).to(torch.long)
                        labels_hm = preds_rescale[:, 0, :]

                    row = [video_name,ts1, ts2, index_video.item(), global_index.item(), partition, h5_index]
                    #Store for each gaze size in config
                    for f_pos, gs in enumerate(cfg.GENERATE.GAZE_SIZE):
                        #Crop and modifiy gaze location depending on overlaps on boundaries
                        gaze_frame, gaze_loc = crop_gaze_image(orig_frame,torch.clone(lab), gs)
                        row.extend([gaze_loc[0].item(), gaze_loc[1].item()])
                        # if gs > 224 and gs != 540:
                        #     gaze_frame = torch.nn.functional.interpolate(gaze_frame.unsqueeze(0), size=(224,224), mode="bicubic")
                        gaze_frame = (gaze_frame.squeeze().numpy()*255).astype(np.uint8)

                        #Store in h5
                        import cv2
                        def numpy_to_binary(arr):
                            is_success, buffer = cv2.imencode(".jpg", arr)
                            io_buf = io.BytesIO(buffer)
                            return io_buf

                        ar2 = numpy_to_binary(np.einsum('ijk->jki', gaze_frame))
                        frames[f_pos].append(np.asarray(ar2.getvalue()))

                        #Display the crops
                        if cfg.GENERATE.LOG:
                            if not os.path.exists(f"../gym_results/test_images/ego4d/{str(gs)}/"):
                                os.makedirs(f"../gym_results/test_images/ego4d/{str(gs)}/")
                            Image.fromarray(gaze_frame.transpose(1,2,0)).save(f"../gym_results/test_images/ego4d/{str(gs)}/" + str(cpt) + ".png")

                    # Display the saliency
                    if cfg.GENERATE.LOG:
                        pred_rescale = normalize_map(labels_hm[i:i+1,k:k+1].unsqueeze(1))[:,:,0]
                        new_frame = (framest[:,k] * cfg.DATA.STD[0]) + cfg.DATA.MEAN[0]
                        pred_rescale = torch.nn.functional.interpolate(pred_rescale, size=new_frame.size()[1:]).squeeze(0).numpy()

                        new_frame = new_frame.numpy()
                        new_frame = show_cam_on_image(new_frame.transpose(1,2,0), pred_rescale.transpose(1,2,0), use_rgb=True)
                        if not os.path.exists("../gym_results/test_images/ego4d/labels_saliency/"):
                            os.makedirs("../gym_results/test_images/ego4d/labels_saliency/")
                        Image.fromarray(new_frame).save("../gym_results/test_images/ego4d/labels_saliency/" + str(cpt) + ".png")


                    # Save in dataset
                    row.append(dataset_number)
                    csv_writer.writerow(row)
                    cpt += 1
                    h5_index += 1
                    if cpt%50000 == 0:
                        for f_pos, gs2 in enumerate(cfg.GENERATE.GAZE_SIZE):
                            hf.create_dataset(f"images{str(gs2)}_{str(dataset_number)}", data=frames[f_pos])
                            frames[f_pos] = []
                        h5_index = 0
                        dataset_number += 1
            test_meter.iter_tic()


        if len(frames[0]) > 0:
            for i, gs2 in enumerate(cfg.GENERATE.GAZE_SIZE):
                hf.create_dataset(f"images{str(gs2)}_{str(dataset_number)}", data=frames[i])
    dataset_number += 1
    print("finish", cpt, test_loader.dataset.num_videos*cfg.DATA.NUM_FRAMES)
    csv_file.close()
    # test_meter.finalize_metrics()

    return test_meter


def generate_label(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (test_loader.dataset.num_videos % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0)
        # Create meters for multi-view testing.
        test_meter = TestGazeMeter(
            num_videos=test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            num_cls=cfg.MODEL.NUM_CLASSES,
            overall_iters=len(test_loader),
            dataset=cfg.TEST.DATASET
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()

    logger.info("Testing finished!")
