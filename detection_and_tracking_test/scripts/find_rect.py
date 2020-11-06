import numpy as np
import math

def find_rect(pts, lower_limit, higher_limit, range_interval):
    rotation_angle = list(range(lower_limit, higher_limit, range_interval))
    dis = []
    best_rotation_angle = float('inf')
    min_dis = float('inf')
    
    for i in range(len(rotation_angle)):
        theta = rotation_angle[i] * math.pi / 180
        down_vx = math.cos(theta)
        down_vz = math.sin(theta)
        up_vx = math.cos(theta + math.pi / 2)
        up_vz = math.sin(theta + math.pi / 2)
        
        # 点云投影到down直线后的x坐标、z坐标
        if abs(down_vz) > 0.0001:
            pts_projection_down_z = pts[:, 1] * down_vz ** 2 + pts[:, 0] * down_vx * down_vz
            pts_projection_down_x = pts_projection_down_z * down_vx / down_vz
        else:
            pts_projection_down_z = 0 * np.ones((pts.shape[0]))
            pts_projection_down_x = pts[:, 0]
        
        # down直线存在水平状态，不存在竖直状态
        # down直线上的两个投影点
        down_max_idx = np.where(pts_projection_down_x == pts_projection_down_x.max())[0][0]
        down_max_p = [pts_projection_down_x[down_max_idx], pts_projection_down_z[down_max_idx]]
        down_min_idx = np.where(pts_projection_down_x == pts_projection_down_x.min())[0][0]
        down_min_p = [pts_projection_down_x[down_min_idx], pts_projection_down_z[down_min_idx]]
        
        # 点云投影到up直线后的x坐标、z坐标
        if abs(up_vz) > 0.0001:
            pts_projection_up_z = pts[:, 1] * up_vz ** 2 + pts[:, 0] * up_vx * up_vz
            pts_projection_up_x = pts_projection_up_z * up_vx / up_vz
        else:
            pts_projection_up_z = 0 * np.ones((pts.shape[0]))
            pts_projection_up_x = pts[:, 0]
        
        # up直线存在竖直状态，不存在水平状态
        # up直线上的两个投影点
        up_max_idx = np.where(pts_projection_up_z == pts_projection_up_z.max())[0][0]
        up_max_p = [pts_projection_up_x[up_max_idx], pts_projection_up_z[up_max_idx]]
        up_min_idx = np.where(pts_projection_up_z == pts_projection_up_z.min())[0][0]
        up_min_p = [pts_projection_up_x[up_min_idx], pts_projection_up_z[up_min_idx]]
        
        # 包络矩形的顶点1
        p1 = [down_min_p[0] + up_min_p[0], down_min_p[1] + up_min_p[1]]
        # 包络矩形的顶点2
        p2 = [down_max_p[0] + up_min_p[0], down_max_p[1] + up_min_p[1]]
        # 包络矩形的顶点3
        p3 = [down_max_p[0] + up_max_p[0], down_max_p[1] + up_max_p[1]]
        # 包络矩形的顶点4
        p4 = [down_min_p[0] + up_max_p[0], down_min_p[1] + up_max_p[1]]
        
        # 点到p1p2的距离
        vx = p2[0] - p1[0]
        vz = p2[1] - p1[1]
        if abs(vx) > 0.0001:
            k = vz / vx
            pts_dis_to_12 = abs(k*pts[:, 0] - pts[:, 1] - k*p1[0] + p1[1]) / np.sqrt(k**2 + 1)
        else:
            pts_dis_to_12 = abs(pts[:, 0] - p1[0])
        # 点到p2p3的距离
        vx = p3[0] - p2[0]
        vz = p3[1] - p2[1]
        if abs(vx) > 0.0001:
            k = vz / vx
            pts_dis_to_23 = abs(k*pts[:, 0] - pts[:, 1] - k*p2[0] + p2[1]) / np.sqrt(k**2 + 1)
        else:
            pts_dis_to_23 = abs(pts[:, 0] - p2[0])
        # 点到p3p4的距离
        vx = p4[0] - p3[0]
        vz = p4[1] - p3[1]
        if abs(vx) > 0.0001:
            k = vz / vx
            pts_dis_to_34 = abs(k*pts[:, 0] - pts[:, 1] - k*p3[0] + p3[1]) / np.sqrt(k**2 + 1)
        else:
            pts_dis_to_34 = abs(pts[:, 0] - p3[0])
        # 点到p4p1的距离
        vx = p1[0] - p4[0]
        vz = p1[1] - p4[1]
        if abs(vx) > 0.0001:
            k = vz / vx
            pts_dis_to_41 = abs(k*pts[:, 0] - pts[:, 1] - k*p4[0] + p4[1]) / np.sqrt(k**2 + 1)
        else:
            pts_dis_to_41 = abs(pts[:, 0] - p4[0])
        
        # 点到矩形边界的距离和
        pts_dis_to_boundary = np.array([pts_dis_to_12, pts_dis_to_23, pts_dis_to_34, pts_dis_to_41]).T
        pts_dis_to_boundary_sum = pts_dis_to_boundary.min(axis=1).sum(axis=0)
        
        dis.append(pts_dis_to_boundary_sum)
        if pts_dis_to_boundary_sum < min_dis:
            min_dis = pts_dis_to_boundary_sum
            best_rotation_angle = rotation_angle[i]
            best_rect = np.array([p1, p2, p3, p4])

    return rotation_angle, dis, best_rotation_angle, best_rect
