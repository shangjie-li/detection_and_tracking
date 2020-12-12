# -*- coding: UTF-8 -*- 

import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time
import math
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import torch
import torch.backends.cudnn as cudnn
import argparse
from collections import defaultdict

from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess
from data import cfg, set_cfg

from sklearn.cluster import DBSCAN

from calib import Calib
from numpy_pc2 import pointcloud2_to_xyz_array
from find_rect import find_rect
from akf_tracker import AugmentKalmanFilter
from jet_color import Jet_Color

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
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--score_threshold', default=0.05, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')

    parser.set_defaults(mask_proto_debug=False, crop=True)

    global args
    args = parser.parse_args(argv)

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
        
    return frame, num_dets_to_consider, target_masks, target_classes

def fusion(picamera, piimage, num_target, target_masks, target_classes):
    target_masks = target_masks.byte().cpu().numpy()
    
    # 输入跟踪器的观测值
    z = np.matrix([])
    z_cube = np.matrix([])
    z_class = []
    
    # 目标点云俯视图包络矩形
    points_rect_set = np.zeros((num_target, 4, 2), dtype=np.float32)
    points_x_set = []
    points_y_set = []
    points_z_set = []
    
    for i in range(num_target):
        points_x = []
        points_y = []
        points_z = []
        
        # 点云与目标实例掩膜匹配
        for pt in range(picamera.shape[0]):
            if target_masks[i, int(piimage[pt][1]), int(piimage[pt][0])]:
                points_x.append(picamera[pt][0])
                points_y.append(picamera[pt][1])
                points_z.append(picamera[pt][2])
        
        if len(points_x):
            pts_xyz = np.array([points_x, points_y, points_z], dtype=np.float32).T
            
            # DBSCAN(Density-Based Spatial Clustering of Applications with Noise)基于密度的聚类算法
            # 将具有足够密度的点云划分为簇，无需给定聚类中心的数量
            # DBSCAN类输入的参数：
            #   eps: float, default=0.5 将两点视为同类的最大距离，该值不决定类中最远的两个点的距离
            #   min_samples: int, default=5 最少聚类点数
            #   metric: string, or callable, default=’euclidean’ 计算距离时使用的度量方式，一般情况下无需设置
            #   metric_params: dict, default=None 度量的关键参数，一般情况下无需设置
            #   algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’ 查找临近点的算法，一般情况下无需设置
            #   leaf_size: int, default=30 传递给ball_tree或kd_tree的参数，会影响查找的速度
            #   p: float, default=None 用于Minkowski度量，一般情况下无需设置
            #   n_jobs: int, default=None 并行运行，一般情况下无需设置
            # DBSCAN类保存的结果：
            #   db.core_sample_indices_存储所有被聚类点的索引
            #   db.components_存储所有被聚类点的坐标
            #   db.labels_存储所有点所在的聚类中心ID，其中噪点为-1
            
            # 聚类密度：在0.5m范围内，点云数量超过5个
            # 当目标点云数量为500时，聚类过程耗时约3ms
            db = DBSCAN(eps=0.5, min_samples=5, leaf_size=30).fit(pts_xyz[:, [0, 2]])
            
            # 若无聚类结果，则退出
            if db.labels_.max() < 0:
                pass
            else:
                # 滤除离群点，并保留最大的聚类簇
                best_label = -1
                num_best_label = 0
                for label in range(db.labels_.max() + 1):
                    num_label = np.where(db.labels_ == label)[0].shape[0]
                    if num_label > num_best_label:
                        num_best_label = num_label
                        best_label = label
                pts_xyz = pts_xyz[np.where(db.labels_ == best_label)[0]]
            
                points_x_set += list(pts_xyz[:, 0])
                points_y_set += list(pts_xyz[:, 1])
                points_z_set += list(pts_xyz[:, 2])
                
                # 根据目标点云中最近点的距离筛选
                pts_dis = np.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 2] ** 2)
                if pts_dis.min() > distance_limit:
                    pass
                else:
                    # 以最近点作为参考点
                    nearest_idx = np.where(pts_dis == pts_dis.min())[0][0]
                    
                    # 行人目标包络立方体
                    if cfg.dataset.class_names[target_classes[i]] in ['person']:
                        b_x = pts_xyz[nearest_idx, 0]
                        b_z = pts_xyz[nearest_idx, 2]
                        rect = np.array([[b_x - 0.25, b_z - 0.25],
                                         [b_x + 0.25, b_z - 0.25],
                                         [b_x + 0.25, b_z + 0.25],
                                         [b_x - 0.25, b_z + 0.25],])
                    
                    # 车辆目标包络立方体
                    elif cfg.dataset.class_names[target_classes[i]] in ['car']:
                        # 以较低精度拟合矩形模型
                        lower_limit = 0
                        higher_limit = 90
                        range_interval = 10
                        rotation_angle, dis, _, _ = find_rect(pts_xyz[:, [0, 2]], lower_limit, higher_limit, range_interval)
                        
                        # 以较高精度拟合矩形模型
                        range_interval = 2
                        min_idx = dis.index(min(dis))
                        if min_idx == 0:
                            _, dis_1, _, best_rect_1 = find_rect(pts_xyz[:, [0, 2]], lower_limit, rotation_angle[1], range_interval)
                            _, dis_2, _, best_rect_2 = find_rect(pts_xyz[:, [0, 2]], rotation_angle[-1], higher_limit, range_interval)
                        elif min_idx == len(dis) - 1:
                            _, dis_1, _, best_rect_1 = find_rect(pts_xyz[:, [0, 2]], rotation_angle[-2], rotation_angle[-1], range_interval)
                            _, dis_2, _, best_rect_2 = find_rect(pts_xyz[:, [0, 2]], rotation_angle[-1], higher_limit, range_interval)
                        else:
                            _, dis_1, _, best_rect_1 = find_rect(pts_xyz[:, [0, 2]], rotation_angle[min_idx - 1], rotation_angle[min_idx], range_interval)
                            _, dis_2, _, best_rect_2 = find_rect(pts_xyz[:, [0, 2]], rotation_angle[min_idx], rotation_angle[min_idx + 1], range_interval)
                        
                        if min(dis_1) < min(dis_2):
                            rect = best_rect_1
                        else:
                            rect = best_rect_2
                        
                    # 记录最小包络立方体各顶点坐标(x,y,z)，相机坐标系中x轴朝右，y轴朝下，z轴朝前
                    points_rect_set[i, :, :] = rect
                    cube_new = np.zeros((8, 3))
                    for p in range(0, 4):
                        cube_new[p, :] = np.array([rect[p, 0], pts_xyz[:, 1].max(), rect[p, 1]], dtype=np.float32)
                    for p in range(4, 8):
                        cube_new[p, :] = np.array([rect[p - 4, 0], pts_xyz[:, 1].min(), rect[p - 4, 1]], dtype=np.float32)
                    
                    # 观测值
                    if z.shape[1] == 0:
                        z = np.matrix([[pts_xyz[nearest_idx, 0]], [pts_xyz[nearest_idx, 2]],])
                        z_cube = cube_new
                        z_class = [cfg.dataset.class_names[target_classes[i]]]
                    else:
                        z = np.column_stack((z, np.matrix([[pts_xyz[nearest_idx, 0]], [pts_xyz[nearest_idx, 2]],])))
                        z_cube = np.column_stack((z_cube, cube_new)) 
                        z_class.append(cfg.dataset.class_names[target_classes[i]])
    
    return z, z_cube, z_class, points_rect_set, points_x_set, points_y_set, points_z_set

def tracking(z, z_cube, z_class):
    # 跟踪器迭代
    tracker.kf_iterate(z, z_cube, z_class)
    xx = tracker.xx
    xx_cube = tracker.xx_cube
    xx_class = tracker.xx_class
    xx_color_idx = tracker.xx_color_idx
    
    return xx, xx_cube, xx_class, xx_color_idx

def get_color(color_idx, on_gpu=None):
    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    
    # Color_cache.
    color_cache = defaultdict(lambda: {})
    
    if on_gpu is not None and color_idx in color_cache[on_gpu]:
        return color_cache[on_gpu][color_idx]
    else:
        color = COLORS[color_idx]
        # The image might come in as RGB or BRG, depending
        if on_gpu is not None:
            color = torch.Tensor(color).to(on_gpu).float() / 255.
            color_cache[on_gpu][color_idx] = color
        return color

def mask_display(img, masks, num_target):
    img_gpu = img / 255.0
    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_target > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        color_idx = 0
        colors = torch.cat([get_color(color_idx, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_target)], dim=0)
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
    
    return img_numpy

def pointcloud_display(img_numpy, camera_xyz, camera_uv):
    jc = Jet_Color()
    depth = np.sqrt(np.square(camera_xyz[:, 0]) + np.square(camera_xyz[:, 1]) + np.square(camera_xyz[:, 2]))
    for pt in range(0, camera_uv.shape[0]):
        cv_color = jc.get_jet_color(depth[pt] * jet_color)
        cv2.circle(img_numpy, (int(camera_uv[pt][0]), int(camera_uv[pt][1])), 1, cv_color, thickness=-1)
    
    return img_numpy

def result_display(img_numpy, xx, xx_cube, xx_class, xx_color_idx):
    if xx.shape[1] == 0:
        return img_numpy
    num_xx = int(len(xx) / 4)
    
    # 按距离从远到近，对包络立方体进行排序
    target_dd = []
    for i in range(num_xx):
        dd = xx[4 * i, 0] ** 2 + xx[4 * i + 2, 0] ** 2
        target_dd.append(dd)
    target_idxs = list(np.argsort(target_dd))
    xx_cube_idxs = []
    for i in range(num_xx):
        for cube_i in range(8):
            xx_cube_idxs.append(8 * target_idxs[i] + cube_i)
    # xx_cube_sorted为8n行3列坐标矩阵(x,y,z)，n为目标数量，每8行中，前4行为底面，后4行为顶面
    xx_cube_sorted = xx_cube[xx_cube_idxs]
    
    # 显示每个立方体
    for i in range(num_xx):
        color = COLORS[xx_color_idx[i]]
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.4
        font_thickness = 1
        
        cube = xx_cube_sorted[range(8 * i, 8 * i + 8), :]
        center_x = np.mean(cube[:, 0])
        center_z = np.mean(cube[:, 2])
        
        # 按照俯视图逆时针，对包络矩形的顶点进行排序
        local_polar_angle = []
        for cube_i in range(4):
            # math.atan2(y, x)返回值范围(-pi, pi]
            local_polar_x = cube[cube_i, 0] - center_x
            local_polar_z = cube[cube_i, 2] - center_z
            local_polar_angle_i = math.atan2(local_polar_z, local_polar_x)
            local_polar_angle.append(local_polar_angle_i)
        local_polar_idxs = list(np.argsort(local_polar_angle))
        cube_idxs = local_polar_idxs + [x + 4 for x in local_polar_idxs]
        # cube_sorted为8行3列坐标矩阵(x,y,z)
        cube_sorted = cube[cube_idxs]
        
        cube_sorted_homogeneous = np.column_stack((cube_sorted, np.ones((8, 1))))
        pixel = calib.projection.dot(cube_sorted_homogeneous.T).T
        pixel = np.true_divide(pixel[:, :2], pixel[:, [-1]])
        pixel = pixel.astype(np.int16)
        # cube_xyz_uv为8行5列坐标矩阵(x,y,z,u,v)
        cube_xyz_uv = np.column_stack((cube_sorted, pixel))
        
        # 计算俯视图中包络矩形各顶点相对坐标原点的极角，范围(0,pi)
        general_polar_angle = []
        for cube_i in range(4):
            # math.atan2(y, x)返回值范围(-pi, pi]
            general_polar_x = cube_xyz_uv[cube_i, 0]
            general_polar_z = cube_xyz_uv[cube_i, 2]
            general_polar_angle_i = math.atan2(general_polar_z, general_polar_x)
            general_polar_angle.append(general_polar_angle_i)
        general_polar_angle = general_polar_angle + general_polar_angle
        # cube_xyz_uv为8行6列坐标矩阵(x,y,z,u,v,polar_angle)
        cube_xyz_uv_polar = np.column_stack((cube_xyz_uv, np.array(general_polar_angle)))
        
        # 判断最小极角的顶点和最大极角的顶点是否为相邻点，只考虑底面的四个顶点
        # 如果是，则能看到包络立方体的一个侧面，如果不是，则能看到包络立方体的两个侧面
        cube_polar = cube_xyz_uv_polar[range(4), 5]
        cube_polar_min_idx = np.where(cube_polar == cube_polar.min())[0][0]
        cube_polar_max_idx = np.where(cube_polar == cube_polar.max())[0][0]
        cube_polar_other_idx = []
        for cube_i in range(4):
            if cube_i in [cube_polar_min_idx, cube_polar_max_idx]:
                continue
            cube_polar_other_idx.append(cube_i)
        pp = abs(cube_polar_max_idx - cube_polar_min_idx)
        # uv为8行2列坐标矩阵(u,v)
        uv = cube_xyz_uv_polar[:, [3, 4]].astype(np.int16)
        
        # 如果能看到包络立方体的两个侧面
        if pp == 2:
            # 寻找最近的顶点
            idx_1 = cube_polar_other_idx[0]
            dd_1 = cube_xyz_uv_polar[idx_1, 0] ** 2 + cube_xyz_uv_polar[idx_1, 2] ** 2
            idx_2 = cube_polar_other_idx[1]
            dd_2 = cube_xyz_uv_polar[idx_2, 0] ** 2 + cube_xyz_uv_polar[idx_2, 2] ** 2
            if dd_1 < dd_2:
                cube_nearest_idx = idx_1
                cube_other_idx = idx_2
            else:
                cube_nearest_idx = idx_2
                cube_other_idx = idx_1
            
            # 判断是否能看到顶面，通过计算其余顶点在图像中的高度，考虑顶面
            # polar_min顶点在图像右侧，polar_max顶点在图像左侧，nearest顶点在中间
            top_can_be_seen = False
            cube_polar_min_u = uv[cube_polar_min_idx + 4, 0]
            cube_polar_min_v = uv[cube_polar_min_idx + 4, 1]
            cube_polar_max_u = uv[cube_polar_max_idx + 4, 0]
            cube_polar_max_v = uv[cube_polar_max_idx + 4, 1]
            cube_other_u = uv[cube_other_idx + 4, 0]
            cube_other_v = uv[cube_other_idx + 4, 1]
            
            k = (cube_polar_min_v - cube_polar_max_v) / (cube_polar_min_u - cube_polar_max_u)
            if cube_other_v < (cube_other_u - cube_polar_max_u) * k + cube_polar_max_v:
                top_can_be_seen = True
            
            # 绘制包络立方体的顶面
            if top_can_be_seen:
                pt_1 = (uv[cube_other_idx + 4, 0], uv[cube_other_idx + 4, 1])
                pt_2 = (uv[cube_polar_max_idx + 4, 0], uv[cube_polar_max_idx + 4, 1])
                cv2.line(img_numpy, pt_1, pt_2, color, 1)
                pt_1 = (uv[cube_other_idx + 4, 0], uv[cube_other_idx + 4, 1])
                pt_2 = (uv[cube_polar_min_idx + 4, 0], uv[cube_polar_min_idx + 4, 1])
                cv2.line(img_numpy, pt_1, pt_2, color, 1)
            
            # 绘制包络立方体的两个侧面
            pt_1 = (uv[cube_nearest_idx, 0], uv[cube_nearest_idx, 1])
            pt_2 = (uv[cube_nearest_idx + 4, 0], uv[cube_nearest_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_polar_max_idx, 0], uv[cube_polar_max_idx, 1])
            pt_2 = (uv[cube_polar_max_idx + 4, 0], uv[cube_polar_max_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_polar_min_idx, 0], uv[cube_polar_min_idx, 1])
            pt_2 = (uv[cube_polar_min_idx + 4, 0], uv[cube_polar_min_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_nearest_idx, 0], uv[cube_nearest_idx, 1])
            pt_2 = (uv[cube_polar_max_idx, 0], uv[cube_polar_max_idx, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_nearest_idx, 0], uv[cube_nearest_idx, 1])
            pt_2 = (uv[cube_polar_min_idx, 0], uv[cube_polar_min_idx, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_nearest_idx + 4, 0], uv[cube_nearest_idx + 4, 1])
            pt_2 = (uv[cube_polar_max_idx + 4, 0], uv[cube_polar_max_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_nearest_idx + 4, 0], uv[cube_nearest_idx + 4, 1])
            pt_2 = (uv[cube_polar_min_idx + 4, 0], uv[cube_polar_min_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            
            # (x1,y1)处显示xx，(x2,y2)处显示xx_class
            x1 = uv[cube_nearest_idx, 0]
            y1 = uv[cube_nearest_idx, 1]
            x2 = uv[cube_nearest_idx + 4, 0]
            y2 = uv[cube_nearest_idx + 4, 1]
            
            text_str = xx_class[i]
            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(img_numpy, (x2, y2), (x2 + text_w, y2 - text_h - 4), color, -1)
            # cv2.putText(图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度)
            cv2.putText(img_numpy, text_str, (x2, y2 - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            text_lo = "(%.1fm, %.1fm)" % (xx[4 * i, 0], xx[4 * i + 2, 0])
            text_w_lo, text_h_lo = cv2.getTextSize(text_lo, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w_lo, y1 + text_h + 4), color, -1)
            # cv2.putText(图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度)
            cv2.putText(img_numpy, text_lo, (x1, y1 + text_h + 1), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            text_ve = "(%.1fm/s, %.1fm/s)" % (xx[4 * i + 1, 0], xx[4 * i + 3, 0])
            text_w_ve, text_h_ve = cv2.getTextSize(text_ve, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(img_numpy, (x1, y1 + text_h + 4), (x1 + text_w_ve, y1 + 2 * text_h + 8), color, -1)
            # cv2.putText(图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度)
            cv2.putText(img_numpy, text_ve, (x1, y1 + 2 * text_h + 5), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # 如果能看到包络立方体的一个侧面
        else:
            # 判断是否能看到顶面，通过计算其余顶点在图像中的高度，考虑顶面
            # polar_min顶点在图像右侧，polar_max顶点在图像左侧
            top_can_be_seen = False
            cube_polar_min_u = uv[cube_polar_min_idx + 4, 0]
            cube_polar_min_v = uv[cube_polar_min_idx + 4, 1]
            cube_polar_max_u = uv[cube_polar_max_idx + 4, 0]
            cube_polar_max_v = uv[cube_polar_max_idx + 4, 1]
            
            k = (cube_polar_min_v - cube_polar_max_v) / (cube_polar_min_u - cube_polar_max_u)
            for cube_i in range(2):
                cube_other_idx = cube_polar_other_idx[cube_i]
                cube_other_u = uv[cube_other_idx + 4, 0]
                cube_other_v = uv[cube_other_idx + 4, 1]
                if cube_other_v < (cube_other_u - cube_polar_max_u) * k + cube_polar_max_v:
                    top_can_be_seen = True
            
            # 绘制包络立方体的顶面
            if top_can_be_seen:
                pt_1 = (uv[4, 0], uv[4, 1])
                pt_2 = (uv[5, 0], uv[5, 1])
                cv2.line(img_numpy, pt_1, pt_2, color, 1)
                pt_1 = (uv[5, 0], uv[5, 1])
                pt_2 = (uv[6, 0], uv[6, 1])
                cv2.line(img_numpy, pt_1, pt_2, color, 1)
                pt_1 = (uv[6, 0], uv[6, 1])
                pt_2 = (uv[7, 0], uv[7, 1])
                cv2.line(img_numpy, pt_1, pt_2, color, 1)
                pt_1 = (uv[7, 0], uv[7, 1])
                pt_2 = (uv[4, 0], uv[4, 1])
                cv2.line(img_numpy, pt_1, pt_2, color, 1)
                
            # 绘制包络立方体的一个侧面
            pt_1 = (uv[cube_polar_max_idx, 0], uv[cube_polar_max_idx, 1])
            pt_2 = (uv[cube_polar_max_idx + 4, 0], uv[cube_polar_max_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_polar_min_idx, 0], uv[cube_polar_min_idx, 1])
            pt_2 = (uv[cube_polar_min_idx + 4, 0], uv[cube_polar_min_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_polar_max_idx, 0], uv[cube_polar_max_idx, 1])
            pt_2 = (uv[cube_polar_min_idx, 0], uv[cube_polar_min_idx, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            pt_1 = (uv[cube_polar_max_idx + 4, 0], uv[cube_polar_max_idx + 4, 1])
            pt_2 = (uv[cube_polar_min_idx + 4, 0], uv[cube_polar_min_idx + 4, 1])
            cv2.line(img_numpy, pt_1, pt_2, color, 1)
            
            # (x1,y1)处显示xx，(x2,y2)处显示xx_class
            x1 = uv[cube_polar_max_idx, 0]
            y1 = uv[cube_polar_max_idx, 1]
            x2 = uv[cube_polar_max_idx + 4, 0]
            y2 = uv[cube_polar_max_idx + 4, 1]
            
            text_str = xx_class[i]
            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(img_numpy, (x2, y2), (x2 + text_w, y2 - text_h - 4), color, -1)
            # cv2.putText(图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度)
            cv2.putText(img_numpy, text_str, (x2, y2 - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            text_lo = "(%.1fm, %.1fm)" % (xx[4 * i, 0], xx[4 * i + 2, 0])
            text_w_lo, text_h_lo = cv2.getTextSize(text_lo, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w_lo, y1 + text_h + 4), color, -1)
            # cv2.putText(图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度)
            cv2.putText(img_numpy, text_lo, (x1, y1 + text_h + 1), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            text_ve = "(%.1fm/s, %.1fm/s)" % (xx[4 * i + 1, 0], xx[4 * i + 3, 0])
            text_w_ve, text_h_ve = cv2.getTextSize(text_ve, font_face, font_scale, font_thickness)[0]
            cv2.rectangle(img_numpy, (x1, y1 + text_h + 4), (x1 + text_w_ve, y1 + 2 * text_h + 8), color, -1)
            # cv2.putText(图像，文字内容，文字左下角所在uv坐标，字体，大小，颜色，字体宽度)
            cv2.putText(img_numpy, text_ve, (x1, y1 + 2 * text_h + 5), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img_numpy

def top_view_result_display(top_view_image, z, points_rect_set, points_x_set, points_y_set, points_z_set):
    fx = 10
    fy = -10
    top_view_image_h = top_view_image.shape[0]
    top_view_image_w = top_view_image.shape[1]
    cx = int(top_view_image_w / 2)
    cy = int(top_view_image_h)
    
    for i in range(top_view_image_h - 20, top_view_image_h):
        top_view_u = top_view_image_w / 2
        top_view_v = i
        cv2.circle(top_view_image, (int(top_view_u), int(top_view_v)), 1, (0, 0, 255), thickness = -1)
    for i in range(int(top_view_image_w / 2), int(top_view_image_w / 2 + 20)):
        top_view_u = i
        top_view_v = top_view_image_h - 2
        cv2.circle(top_view_image, (int(top_view_u), int(top_view_v)), 1, (0, 0, 255), thickness = -1)
    
    for i in range(top_view_image_h):
        top_view_u = 0
        top_view_v = i
        cv2.circle(top_view_image, (top_view_u, top_view_v), 1, (255, 255, 255), thickness = -1)
        top_view_u = top_view_image_w
        top_view_v = i
        cv2.circle(top_view_image, (top_view_u, top_view_v), 1, (255, 255, 255), thickness = -1)
    for i in range(top_view_image_w):
        top_view_u = i
        top_view_v = 0
        cv2.circle(top_view_image, (top_view_u, top_view_v), 1, (255, 255, 255), thickness = -1)
        top_view_u = i
        top_view_v = top_view_image_h - 1
        cv2.circle(top_view_image, (top_view_u, top_view_v), 1, (255, 255, 255), thickness = -1)
    
    for i in range(points_rect_set.shape[0]):
        if points_rect_set[i, 0, 0] and points_rect_set[i, 0, 1]:
            for p in range(4):
                pt_1 = (int(fx * points_rect_set[i, p, 0] + cx), int(fy * points_rect_set[i, p, 1] + cy))
                pt_2 = (int(fx * points_rect_set[i, (p + 1)%4, 0] + cx), int(fy * points_rect_set[i, (p + 1)%4, 1] + cy))
                cv2.line(top_view_image, pt_1, pt_2, (0, 255, 255), 1)
    
    targets_x_z = np.array([points_x_set,
                            points_z_set], dtype=np.float32)
    if z.shape[1] == 0:
        pass
    else:
        for targets_pt in range(targets_x_z.shape[1]):
            top_view_u = int(fx * targets_x_z[0, targets_pt] + cx)
            top_view_v = int(fy * targets_x_z[1, targets_pt] + cy)
            if top_view_v >= 0 and top_view_v < top_view_image_h and top_view_u >= 0 and top_view_u < top_view_image_w:
                # thickness正值表示圆形轮廓的粗细，负值表示绘制实心圆
                cv2.circle(top_view_image, (top_view_u, top_view_v), 1, (255, 255, 255), thickness = -1)
        for targets_pt in range(z.shape[1]):
            top_view_u = int(fx * z[0, targets_pt] + cx)
            top_view_v = int(fy * z[1, targets_pt] + cy)
            if top_view_v >= 0 and top_view_v < top_view_image_h and top_view_u >= 0 and top_view_u < top_view_image_w:
                # thickness正值表示圆形轮廓的粗细，负值表示绘制实心圆
                cv2.circle(top_view_image, (top_view_u, top_view_v), 1, (0, 0, 255), thickness = -1)
    
    return top_view_image

def convert(header, xx_cube):
    boundingboxarray_msg = BoundingBoxArray()
    boundingboxarray_msg.header = header
    
    num_xx = int(xx_cube.shape[0] / 8)
    for i in range(num_xx):
        cube_raw = xx_cube[range(8 * i, 8 * i + 8), :]
        bb_in_camera = np.column_stack((cube_raw, np.ones((8, 1))))
        bb_in_lidar = np.linalg.inv(calib.lidar_to_cam).dot(bb_in_camera.T).T
        boundingbox_msg = BoundingBox()
        boundingbox_msg.header = boundingboxarray_msg.header
        
        # boundingbox中心点位置
        boundingbox_msg.pose.position.x = bb_in_lidar[:, 0].mean()
        boundingbox_msg.pose.position.y = bb_in_lidar[:, 1].mean()
        boundingbox_msg.pose.position.z = bb_in_lidar[:, 2].mean()
        
        # 寻找y坐标最小的顶点，计算相邻两个顶点的旋转角及边长
        bb_bottom = bb_in_lidar[:4]
        min_idx = np.where(bb_bottom[:, 1] == bb_bottom[:, 1].min())[0][0]
        theta = math.atan2(bb_bottom[(min_idx + 1)%4, 1] - bb_bottom[min_idx, 1], bb_bottom[(min_idx + 1)%4, 0] - bb_bottom[min_idx, 0])
        b_1 = ((bb_bottom[(min_idx + 1)%4, 1] - bb_bottom[min_idx, 1]) ** 2 + (bb_bottom[(min_idx + 1)%4, 0] - bb_bottom[min_idx, 0]) ** 2) ** 0.5
        b_2 = ((bb_bottom[(min_idx + 3)%4, 1] - bb_bottom[min_idx, 1]) ** 2 + (bb_bottom[(min_idx + 3)%4, 0] - bb_bottom[min_idx, 0]) ** 2) ** 0.5
        if theta < 90 * math.pi / 180:
            rotation_angle = theta
            dimension_x = b_1
            dimension_y = b_2
        else:
            rotation_angle = theta - 90 * math.pi / 180
            dimension_x = b_2
            dimension_y = b_1
        
        # boundingbox旋转角四元数
        boundingbox_msg.pose.orientation.x = 0
        boundingbox_msg.pose.orientation.y = 0
        boundingbox_msg.pose.orientation.z = math.sin(0.5 * rotation_angle)
        boundingbox_msg.pose.orientation.w = math.cos(0.5 * rotation_angle)
        
        # boundingbox尺寸
        boundingbox_msg.dimensions.x = dimension_x
        boundingbox_msg.dimensions.y = dimension_y
        boundingbox_msg.dimensions.z = bb_in_lidar[:, 2].max() - bb_in_lidar[:, 2].min()
        
        boundingbox_msg.value = 0
        boundingbox_msg.label = 0
        boundingboxarray_msg.boxes.append(boundingbox_msg)
    
    return boundingboxarray_msg

def image_callback(image):
    global image_stamp_list
    global cv_image_list
    
    image_stamp = image.header.stamp.secs + 0.000000001 * image.header.stamp.nsecs
    cv_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    
    if len(image_stamp_list) < 30:
        image_stamp_list.append(image_stamp)
        cv_image_list.append(cv_image)
    else:
        image_stamp_list.pop(0)
        cv_image_list.pop(0)
        image_stamp_list.append(image_stamp)
        cv_image_list.append(cv_image)

def velodyne_callback(pointcloud):
    time_start_all = time.time()
    
    global image_stamp_list
    global cv_image_list
    global detection_missing
    
    # 相机与激光雷达消息同步
    lidar_stamp = pointcloud.header.stamp.secs + 0.000000001 * pointcloud.header.stamp.nsecs
    et_m = float('inf')
    id_stamp = 0
    for t in range(len(image_stamp_list)):
        et = abs(image_stamp_list[t] - lidar_stamp)
        if et < et_m:
            et_m = et
            id_stamp = t
    cv_image = cv_image_list[id_stamp]
    
    # 动态控制主窗口显示
    global display_main_window
    display_main_window = rospy.get_param("~display_main_window_mode")
    global record_main_window
    record_main_window = rospy.get_param("~record_main_window_mode")
    global main_window_initialized
    global video_main
    
    if not display_main_window:
        try:
            cv2.destroyWindow("main")
        except:
            pass
    
    # 动态控制主窗口记录
    if record_main_window and not main_window_initialized:
        record_path = 'main.mp4'
        target_fps = frame_rate
        frame_height = cv_image.shape[0]
        frame_width = cv_image.shape[1]
        video_main = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height), True)
        main_window_initialized = True
        print("Start recording of main window.")
    
    if not record_main_window and main_window_initialized:
        video_main.release()
        main_window_initialized = False
        print("Save video of main window.")
    
    # 动态控制俯视窗口显示
    global display_top_view_window
    display_top_view_window = rospy.get_param("~display_top_view_window_mode")
    global record_top_view_window
    record_top_view_window = rospy.get_param("~record_top_view_window_mode")
    global top_view_window_initialized
    global video_top_view
    
    if not display_top_view_window:
        try:
            cv2.destroyWindow("top_view")
        except:
            pass
    
    # 动态控制俯视窗口记录
    if record_top_view_window and not top_view_window_initialized:
        record_path = 'top_view.mp4'
        target_fps = frame_rate
        frame_height = 500
        frame_width = 400
        video_top_view = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height), True)
        top_view_window_initialized = True
        print("Start recording of top view window.")
    
    if not record_top_view_window and top_view_window_initialized:
        video_top_view.release()
        top_view_window_initialized = False
        print("Save video of top view window.")
    
    # 动态控制终端输出信息
    global print_time_stamp
    print_time_stamp = rospy.get_param("~print_time_stamp_mode")
    global print_xx
    print_xx = rospy.get_param("~print_xx_mode")
    
    global display_mask
    display_mask = rospy.get_param("~display_mask_mode")
    global display_pointcloud
    display_pointcloud = rospy.get_param("~display_pointcloud_mode")
    global display_result
    display_result = rospy.get_param("~display_result_mode")
    
    # 主窗口图像
    image_frame = cv_image
    result_image = image_frame.copy()
    image_h = image_frame.shape[0]
    image_w = image_frame.shape[1]
    
    # 俯视窗口图像
    top_view_image_h = 500
    top_view_image_w = 400
    top_view_image = np.zeros((top_view_image_h, top_view_image_w, 3), dtype=np.uint8)
    
    # 目标检测
    time_start = time.time()
    current_frame, num_target, target_masks, target_classes = detection(image_frame)
    time_detection = round(time.time() - time_start, 3)
    
    # 若连续多帧未检测到目标，则初始化跟踪器
    if num_target > 0:
        detection_missing = 0
    else:
        detection_missing += 1
        if detection_missing > max_tracking_times:
            tracker.kf_initialize(COLORS)
    
    # 载入激光雷达点云
    pointXYZ = pointcloud2_to_xyz_array(pointcloud, remove_nans=True)
    if is_limit_mode:
        alpha = 90 - 0.5 * the_field_of_view
        k = math.tan(alpha * math.pi / 180.0)
        if the_view_number == 1:
            pointXYZ = pointXYZ[np.logical_and((pointXYZ[:, 0] > k * pointXYZ[:, 1]), (pointXYZ[:, 0] > -k * pointXYZ[:, 1]))]
        elif the_view_number == 2:
            pointXYZ = pointXYZ[np.logical_and((-pointXYZ[:, 1] > k * pointXYZ[:, 0]), (-pointXYZ[:, 1] > -k * pointXYZ[:, 0]))]
        elif the_view_number == 3:
            pointXYZ = pointXYZ[np.logical_and((-pointXYZ[:, 0] > k * pointXYZ[:, 1]), (-pointXYZ[:, 0] > -k * pointXYZ[:, 1]))]
        elif the_view_number == 4:
            pointXYZ = pointXYZ[np.logical_and((pointXYZ[:, 1] > k * pointXYZ[:, 0]), (pointXYZ[:, 1] > -k * pointXYZ[:, 0]))]
    if is_clip_mode:
        pointXYZ = pointXYZ[np.logical_and((pointXYZ[:, 0] ** 2 + pointXYZ[:, 1] ** 2 > the_min_distance ** 2), (pointXYZ[:, 0] ** 2 + pointXYZ[:, 1] ** 2 < the_max_distance ** 2))]
        pointXYZ = pointXYZ[np.logical_and((pointXYZ[:, 2] > the_view_lower_limit - the_sensor_height), (pointXYZ[:, 2] < the_view_higher_limit - the_sensor_height))]
    
    cloud_xyz = calib.lidar_to_cam.dot(pointXYZ.T).T
    cloud_uv = calib.lidar_to_img.dot(pointXYZ.T).T
    cloud_uv = np.true_divide(cloud_uv[:, :2], cloud_uv[:, [-1]])
    camera_xyz = cloud_xyz[(cloud_uv[:, 0] >= 0) & (cloud_uv[:, 0] < image_w) & (cloud_uv[:, 1] >= 0) & (cloud_uv[:, 1] < image_h)]
    camera_uv = cloud_uv[(cloud_uv[:, 0] >= 0) & (cloud_uv[:, 0] < image_w) & (cloud_uv[:, 1] >= 0) & (cloud_uv[:, 1] < image_h)]
    
    # 相机与激光雷达数据融合
    time_start = time.time()
    z, z_cube, z_class, points_rect_set, points_x_set, points_y_set, points_z_set = fusion(camera_xyz, camera_uv, num_target, target_masks, target_classes)
    time_fusion = round(time.time() - time_start, 3)
    
    # 目标跟踪
    time_start = time.time()
    xx, xx_cube, xx_class, xx_color_idx = tracking(z, z_cube, z_class)
    time_tracking = round(time.time() - time_start, 3)
    
    # 输出结果
    boundingboxarray_msg = convert(pointcloud.header, xx_cube)
    pub.publish(boundingboxarray_msg)
    
    # 结果可视化
    time_start = time.time()
    if display_main_window or record_main_window:
        if display_mask:
            result_image = mask_display(current_frame, target_masks, num_target)
        if display_pointcloud:
            result_image = pointcloud_display(result_image, camera_xyz, camera_uv)
        if display_result:
            result_image = result_display(result_image, xx, xx_cube, xx_class, xx_color_idx)
    if display_top_view_window or record_top_view_window:
        top_view_image = top_view_result_display(top_view_image, z, points_rect_set, points_x_set, points_y_set, points_z_set)
    
    # 主窗口显示
    if display_main_window:
        cv2.imshow("main", result_image)
    if record_main_window and main_window_initialized:
        video_main.write(result_image)
    
    # 俯视窗口显示
    if display_top_view_window:
        cv2.imshow("top_view", top_view_image)
    if record_top_view_window and top_view_window_initialized:
        video_top_view.write(top_view_image)
    
    # 显示图像时按Esc键终止程序
    if display_main_window or display_top_view_window:
        if cv2.waitKey(1) == 27:
            if display_main_window:
                cv2.destroyWindow("main")
            if record_main_window:
                video_main.release()
                main_window_initialized = False
                print("Save video of main window.")
            if display_top_view_window:
                cv2.destroyWindow("top_view")
            if record_top_view_window:
                video_top_view.release()
                top_view_window_initialized = False
                print("Save video of top view window.")
            rospy.signal_shutdown("It's over.")
    
    time_display = round(time.time() - time_start, 3)
    time_total = round(time.time() - time_start_all, 3)
    
    if print_time_stamp:
        print()
        print("image_stamp:", image_stamp_list[id_stamp])
        print("lidar_stamp:", lidar_stamp)
        print("detection time cost:", time_detection, "s")
        print("fusion time cost:", time_fusion, "s")
        print("tracking time cost:", time_tracking, "s")
        print("display time cost:", time_display, "s")
        print("total time cost:", time_total, "s")
    
    if print_xx:
        print()
        print("detection_missing:", detection_missing)
        print("xx_mistracking:", tracker.xx_mistracking)
        print("xx:")
        for i in range(int(len(tracker.xx) / 4)):
            print(np.around(tracker.xx[4*i:4*i+4, :], decimals=1))

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
    # 发布消息队列设为1，订阅消息队列设为1，并保证订阅消息缓冲区足够大
    # 这样可以实现每次订阅最新的节点消息，避免因队列消息拥挤而导致的雷达点云延迟
    rospy.init_node("dt")
    
    # 动态控制主窗口显示、记录
    display_main_window = False
    record_main_window = False
    main_window_initialized = False
    video_main = None
    
    # 动态控制俯视窗口显示、记录
    display_top_view_window = False
    record_top_view_window = False
    top_view_window_initialized = False
    video_top_view = None
    
    # 动态控制终端输出信息
    print_time_stamp = False
    print_xx = False
    
    display_mask = False
    display_pointcloud = False
    display_result = False
    
    image_topic_name = rospy.get_param("~image_topic")
    lidar_topic_name = rospy.get_param("~lidar_topic")
    pub_topic_name = rospy.get_param("~pub_topic")
    
    calib = Calib()
    file_path = rospy.get_param("~calibration_file_path")
    calib.loadcalib(file_path)
    
    is_limit_mode = rospy.get_param("~is_limit_mode")
    the_view_number = rospy.get_param("~the_view_number")
    the_field_of_view = rospy.get_param("~the_field_of_view")
    
    is_clip_mode = rospy.get_param("~is_clip_mode")
    the_sensor_height = rospy.get_param("~the_sensor_height")
    the_view_higher_limit = rospy.get_param("~the_view_higher_limit")
    the_view_lower_limit = rospy.get_param("~the_view_lower_limit")
    the_min_distance = rospy.get_param("~the_min_distance")
    the_max_distance = rospy.get_param("~the_max_distance")
    
    top_k_person = rospy.get_param("~top_k_person")
    top_k_vehicle = rospy.get_param("~top_k_vehicle")
    score_threshold_person = rospy.get_param("~score_threshold_person")
    score_threshold_vehicle = rospy.get_param("~score_threshold_vehicle")
    
    distance_limit = rospy.get_param("~distance_limit")
    frame_rate = rospy.get_param("~frame_rate")
    associate_threshold_1 = rospy.get_param("~associate_threshold_person")
    associate_threshold_2 = rospy.get_param("~associate_threshold_vehicle")
    max_tracking_times = rospy.get_param("~max_tracking_times")
    sigmaax = rospy.get_param("~sigmaax")
    sigmaay = rospy.get_param("~sigmaay")
    sigmaox = rospy.get_param("~sigmaox")
    sigmaoy = rospy.get_param("~sigmaoy")
    
    jet_color = rospy.get_param("~jet_color")
    
    # 建立增广卡尔曼滤波跟踪器
    tracker = AugmentKalmanFilter(1/frame_rate, associate_threshold_1, associate_threshold_2, max_tracking_times, sigmaax, sigmaay, sigmaox, sigmaoy)
    # 初始化跟踪器
    tracker.kf_initialize(COLORS)
    
    # 连续丢失目标检测帧数
    detection_missing = 0
    
    print('Waiting for topic...')
    image_stamp_list = []
    cv_image_list = []
    rospy.Subscriber(image_topic_name, Image, image_callback, queue_size=1, buff_size=52428800)
    while len(image_stamp_list) < 30:
        time.sleep(1)
    cv_image = cv_image_list[-1]
    frame = torch.from_numpy(cv_image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    print('  Done.')
    
    rospy.Subscriber(lidar_topic_name, PointCloud2, velodyne_callback, queue_size=1, buff_size=52428800)
    pub = rospy.Publisher(pub_topic_name, BoundingBoxArray, queue_size=1)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
