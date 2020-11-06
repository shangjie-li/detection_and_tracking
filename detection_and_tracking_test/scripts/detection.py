# -*- coding: UTF-8 -*- 

import rospy
from sensor_msgs.msg import Image
import numpy as np
import time
import math

import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import random
import cProfile
import pickle
import json
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess
from data import cfg, set_cfg

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        #~ default='weights/yolact_base_6451_400000.pth', type=str,
                        #~ default='weights/yolact_base_54_800000.pth', type=str,
                        default='weights/yolact_resnet50_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.05, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

def detection(image_frame):
    with torch.no_grad():
        frame = torch.from_numpy(image_frame).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        
        # 建立每个目标的蒙版target_masks、类别target_classes、置信度target_scores、边界框target_boxes
        h, w, _ = frame.shape
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            # 检测结果
            t = postprocess(preds, w, h, visualize_lincomb = args.display_lincomb,
                                         crop_masks        = args.crop,
                                         score_threshold   = args.score_threshold)
            cfg.rescore_bbox = save
        
        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:args.top_k]
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        
        # 提取类别为'person'和'car'的目标
        remain_list = []
        items_1 = ['person']
        items_2 = ['car']
        num_items_1 = 0
        num_items_2 = 0
        for j in range(classes.shape[0]):
            if cfg.dataset.class_names[classes[j]] in items_1:
                if num_items_1 < top_k_person and scores[j] > score_threshold_person:
                    remain_list.append(j)
                    num_items_1 += 1
            elif cfg.dataset.class_names[classes[j]] in items_2:
                if num_items_2 < top_k_vehicle and scores[j] > score_threshold_vehicle:
                    remain_list.append(j)
                    num_items_2 += 1
        num_dets_to_consider = len(remain_list)
        
        target_masks = masks[remain_list]
        target_classes = classes[remain_list]
        target_scores = scores[remain_list]
        target_boxes = boxes[remain_list]
        
    return frame, num_dets_to_consider, target_masks, target_classes, target_scores, target_boxes

# Color_cache.
color_cache = defaultdict(lambda: {})

def get_color(color_idx, on_gpu=None):
    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    global color_cache
    
    if on_gpu is not None and color_idx in color_cache[on_gpu]:
        return color_cache[on_gpu][color_idx]
    else:
        color = COLORS[color_idx]
        # The image might come in as RGB or BRG, depending
        if on_gpu is not None:
            color = torch.Tensor(color).to(on_gpu).float() / 255.
            color_cache[on_gpu][color_idx] = color
        return color

def result_display(img, masks, classes, scores, boxes, num_target):
    img_gpu = img / 255.0
    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_target > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_target)], dim=0)
        mask_alpha = 0.45
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_target):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_target > 1:
            inv_alph_cumul = inv_alph_masks[:(num_target-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    # 按置信度从低到高的顺序显示目标信息
    for i in reversed(range(num_target)):
        color = COLORS[i]
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.4
        font_thickness = 1
        
        # 图像左上角坐标x1 y1
        x1, y1, x2, y2 = boxes[i, :]
        # 显示检测结果
        score = scores[i]
        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
        _class = cfg.dataset.class_names[classes[i]]
        text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        # 图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度
        cv2.putText(img_numpy, text_str, (x1, y1 - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img_numpy

def image_callback(image_data):
    cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
    image_frame = cv_image[:, :, [2, 1, 0]]
    current_frame, num_target, target_masks, target_classes, target_scores, target_boxes = detection(image_frame)
    result_image = result_display(current_frame, target_masks, target_classes, target_scores, target_boxes, num_target)
    
    video_out.write(result_image)
    cv2.imshow("result_image", result_image)
    if cv2.waitKey(1) == 27:
        video_out.release()
        cv2.destroyAllWindows()
        rospy.signal_shutdown("It's over.")

if __name__ == '__main__':
    # 解析网络参数
    parse_args()
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    # 加载网络模型
    print('Loading model...')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    if args.cuda:
        net = net.cuda()
    
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    print('  Done.')
    
    # 初始化节点
    rospy.init_node("detection")
    
    top_k_person = 6
    top_k_vehicle = 8
    score_threshold_person = 0.15
    score_threshold_vehicle = 0.65
    
    # 设置保存结果文件
    out_path = 'video_out.mp4'
    target_fps = 20
    frame_height = 480
    frame_width = 640
    video_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height), True)
    
    print('Waiting for topic...')
    rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback, queue_size=1, buff_size=52428800)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
