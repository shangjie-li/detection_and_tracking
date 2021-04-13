# -*- coding: UTF-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

PI = 3.14159

def transform_2d_point_clouds(xs, ys, phi, x0, y0):
    # 功能：对平面点云进行旋转、平移
    # 输入：xs <class 'numpy.ndarray'> (n,) 代表横坐标
    #      ys <class 'numpy.ndarray'> (n,) 代表纵坐标
    #      phi <class 'float'> 为旋转角度，单位为rad
    #      x0 <class 'float'> 为横轴平移量
    #      y0 <class 'float'> 为纵轴平移量
    # 输出：xst <class 'numpy.ndarray'> (n,) 代表旋转、平移后横坐标
    #      yst <class 'numpy.ndarray'> (n,) 代表旋转、平移后纵坐标
    
    xst = xs * math.cos(phi) - ys * math.sin(phi)
    yst = xs * math.sin(phi) + ys * math.cos(phi)
    
    xst += x0
    yst += y0
    
    return xst, yst

def read_gps_data(filename, xg1, yg1, xg2, yg2, print_mode=True):
    # 功能：读入GPS数据
    
    # 载入文件
    with open(filename) as fob:
        infos = fob.readlines()
    if print_mode:
        print()
        print('reading gps data')
        print(type(infos), len(infos))
        print()
    
    # 初始化GPS数据列表
    time_gps_list = []
    xg_list, yg_list, vxg_list, vyg_list, phig_list = [], [], [], [], []
    
    # 逐行读取
    num_frame = len(infos)
    for i in range(num_frame):
        info = list(infos[i])
        
        # 读取time
        k = 0
        while info[k] != '\t': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time_gps = float(temp_str)
        time_gps_list.append(time_gps)
        info = info[k + 1:]
        
        # 读取xg
        k = 0
        while info[k] != '\t': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        xg = float(temp_str)
        xg_list.append(xg)
        info = info[k + 1:]
        
        # 读取yg
        k = 0
        while info[k] != '\t': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        yg = float(temp_str)
        yg_list.append(yg)
        info = info[k + 1:]
        
        k = 0
        while info[k] != '\n': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        phig = float(temp_str)
        phig_list.append(phig)
        
        if print_mode:
            print('time_gps:%f xg:%f yg:%f phig:%f' % (
                time_gps, xg, yg, phig))
    
    # 坐标变换
    xgs = np.array(xg_list)
    ygs = np.array(yg_list)
    phi0 = math.atan2((yg2 - yg1), (xg2 - xg1))
    
    xgs, ygs = transform_2d_point_clouds(xgs, ygs, 0, -xg1, -yg1)
    xgs, ygs = transform_2d_point_clouds(xgs, ygs, -phi0, 0, 0)
    xg_list = list(xgs)
    yg_list = list(ygs)
    
    for i in range(num_frame):
        # 计算vxg
        if i == 0 or i == 1:
            vxg_list.append(0)
        else:
            vxg = (xg_list[i] - xg_list[i - 2]) / 0.1
            vxg_list.append(vxg)
        
        # 计算vyg
        if i == 0 or i == 1:
            vyg_list.append(0)
        else:
            vyg = (yg_list[i] - yg_list[i - 2]) / 0.1
            vyg_list.append(vyg)
            
        # 变换phig
        phig_list[i] -= phi0
        if phig_list[i] > PI:
            phig_list[i] -= PI
        if phig_list[i] < 0:
            phig_list[i] += PI

    return time_gps_list, xg_list, yg_list, vxg_list, vyg_list, phig_list

def read_result_data(filename, print_mode=True):
    # 功能：读入结果数据
    
    # 载入文件
    with open(filename) as fob:
        infos = fob.readlines()
    if print_mode:
        print()
        print('reading result data')
        print(type(infos), len(infos))
        print()
    
    # 初始化结果列表
    time_result_list, frame_list, id_list, x_list, y_list,  vx_list, vy_list = [], [], [], [], [], [], []
    x0_list, y0_list, z0_list, l_list, w_list, h_list, phi_list = [], [], [], [], [], [], []
    
    # 逐行读取
    num_frame = len(infos)
    for i in range(num_frame):
        info = list(infos[i])
        
        # 读取time
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time = float(temp_str)
        time_result_list.append(time)
        
        # 读取frame
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        frame = int(temp_str)
        frame_list.append(frame)
        
        # 读取id
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        id_number = int(temp_str)
        id_list.append(id_number)
        
        # 读取x
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        x = float(temp_str)
        x_list.append(x)
        
        # 读取vx
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        vx = float(temp_str)
        vx_list.append(vx)
        
        # 读取y
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        y = float(temp_str)
        y_list.append(y)
        
        # 读取vy
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        vy = float(temp_str)
        vy_list.append(vy)
        
        # 读取x0
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        x0 = float(temp_str)
        x0_list.append(x0)
        
        # 读取y0
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        y0 = float(temp_str)
        y0_list.append(y0)
        
        # 读取z0
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        z0 = float(temp_str)
        z0_list.append(z0)
        
        # 读取l
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        l = float(temp_str)
        l_list.append(l)
        
        # 读取w
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        w = float(temp_str)
        w_list.append(w)
        
        # 读取h
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        h = float(temp_str)
        h_list.append(h)
        
        # 读取phi
        while info.pop(0) != ':': pass
        k = len(info)
        temp_str = ''
        for j in range(k): temp_str += info[j]
        phi = float(temp_str)
        phi_list.append(phi)
        
        if print_mode:
            print('time:%f frame:%d id:%d x:%f y:%f vx:%f vy:%f x0:%f y0:%f z0:%f l:%f w:%f h:%f phi:%f' % (
                time, frame, id_number, x, y, vx, vy, x0, y0, z0, l, w, h, phi))

    return time_result_list, frame_list, id_list, x_list, y_list, vx_list, vy_list, \
        x0_list, y0_list, z0_list, l_list, w_list, h_list, phi_list

def compute_error(time_gps_list, time_result_list, xg_list, yg_list, vxg_list, vyg_list, length, width, phig_list,
        x0_list, y0_list, vx_list, vy_list, l_list, w_list, phi_list):
    # 功能：计算误差
    
    ex_list, ey_list, evx_list, evy_list, el_list, ew_list, ephi_list = [], [], [], [], [], [], []
    for i in range(len(time_result_list)):
        time = time_result_list[i]
        idx = 0
        dtm = float('inf')
        for j in range(len(time_gps_list)):
            dt = abs(time - time_gps_list[j])
            if dt < dtm:
                idx = j
                dtm = dt
        
        ex = abs(xg_list[idx] - x0_list[i])
        evx = abs(vxg_list[idx] - vx_list[i])
        ey = abs(yg_list[idx] - y0_list[i])
        evy = abs(vyg_list[idx] - vy_list[i])
        el = abs(length - l_list[i])
        ew = abs(width - w_list[i])
        ephi = abs(phig_list[idx] - phi_list[i])
        if ephi > PI / 2:
            ephi = PI - ephi
        
        ex_list.append(ex)
        evx_list.append(evx)
        ey_list.append(ey)
        evy_list.append(evy)
        el_list.append(el)
        ew_list.append(ew)
        ephi_list.append(ephi)

    return ex_list, ey_list, evx_list, evy_list, el_list, ew_list, ephi_list
    
def crop_data(dis_list, ex_list, ey_list, evx_list, evy_list, el_list, ew_list, ephi_list, min_dis=0, max_dis=50):
    # 功能：截取有效数据
    
    idxs = []
    for i in range(len(dis_list)):
        if dis_list[i] > min_dis and dis_list[i] < max_dis:
            idxs.append(i)
    
    dis_list_temp = dis_list
    ex_list_temp = ex_list
    ey_list_temp = ey_list
    evx_list_temp = evx_list
    evy_list_temp = evy_list
    el_list_temp = el_list
    ew_list_temp = ew_list
    ephi_list_temp = ephi_list
    
    dis_list, ex_list, ey_list, evx_list, evy_list, el_list, ew_list, ephi_list = [], [], [], [], [], [], [], []
    
    for i in range(len(idxs)):
        dis_list.append(dis_list_temp[idxs[i]])
        ex_list.append(ex_list_temp[idxs[i]])
        ey_list.append(ey_list_temp[idxs[i]])
        evx_list.append(evx_list_temp[idxs[i]])
        evy_list.append(evy_list_temp[idxs[i]])
        el_list.append(el_list_temp[idxs[i]])
        ew_list.append(ew_list_temp[idxs[i]])
        ephi_list.append(ephi_list_temp[idxs[i]])
    
    return dis_list, ex_list, ey_list, evx_list, evy_list, el_list, ew_list, ephi_list
    
def plot_data(axes, time_gps_list, time_result_list, xg_list, yg_list, vxg_list, vyg_list, length, width, phig_list,
        x0_list, y0_list, vx_list, vy_list, l_list, w_list, phi_list, color='red'):
    # 功能：绘制数据
    
    # 绘制x
    axes[0, 0].plot(time_gps_list, xg_list, color='black', linestyle='-', linewidth=1, label='GPS')
    axes[0, 0].plot(time_result_list, x0_list, color, linestyle='-', linewidth=1, label='Estimate')
    axes[0, 0].legend(loc='upper right', fontsize=8, ncol=1)
    axes[0, 0].set_title('Location in X - Frame', fontsize=10)
    
    # 绘制y
    axes[0, 1].plot(time_gps_list, yg_list, color='black', linestyle='-', linewidth=1, label='GPS')
    axes[0, 1].plot(time_result_list, y0_list, color, linestyle='-', linewidth=1, label='Estimate')
    axes[0, 1].legend(loc='upper right', fontsize=8, ncol=1)
    axes[0, 1].set_title('Location in Y - Frame', fontsize=10)
    
    # 绘制vx
    axes[1, 0].plot(time_gps_list, vxg_list, color='black', linestyle='-', linewidth=1, label='GPS')
    axes[1, 0].plot(time_result_list, vx_list, color, linestyle='-', linewidth=1, label='Estimate')
    axes[1, 0].legend(loc='upper right', fontsize=8, ncol=1)
    axes[1, 0].set_title('Velocity in X - Frame', fontsize=10)
    
    # 绘制vy
    axes[1, 1].plot(time_gps_list, vyg_list, color='black', linestyle='-', linewidth=1, label='GPS')
    axes[1, 1].plot(time_result_list, vy_list, color, linestyle='-', linewidth=1, label='Estimate')
    axes[1, 1].legend(loc='upper right', fontsize=8, ncol=1)
    axes[1, 1].set_title('Velocity in Y - Frame', fontsize=10)
    
    # 绘制l
    axes[2, 0].plot(time_gps_list, [LENGTH] * len(time_gps_list), color='black', linestyle='-', linewidth=1, label='Truth')
    axes[2, 0].plot(time_result_list, l_list, color, linestyle='-', linewidth=1, label='Estimate')
    axes[2, 0].legend(loc='upper right', fontsize=8, ncol=1)
    axes[2, 0].set_title("Object's length - Frame", fontsize=10)
    
    # 绘制w
    axes[2, 1].plot(time_gps_list, [WIDTH] * len(time_gps_list), color='black', linestyle='-', linewidth=1, label='Truth')
    axes[2, 1].plot(time_result_list, w_list, color, linestyle='-', linewidth=1, label='Estimate')
    axes[2, 1].legend(loc='upper right', fontsize=8, ncol=1)
    axes[2, 1].set_title("Object's width - Frame", fontsize=10)
    
    # 绘制phi
    axes[3, 0].plot(time_gps_list, phig_list, color='black', linestyle='-', linewidth=1, label='GPS')
    axes[3, 0].plot(time_result_list, phi_list, color, linestyle='-', linewidth=1, label='Estimate')
    axes[3, 0].legend(loc='upper right', fontsize=8, ncol=1)
    axes[3, 0].set_title("Object's azimuth - Frame", fontsize=10)
    
    axes[3, 1].spines['right'].set_color('none')
    axes[3, 1].spines['left'].set_color('none')
    axes[3, 1].spines['top'].set_color('none')
    axes[3, 1].spines['bottom'].set_color('none')
    axes[3, 1].set_xticks([])
    axes[3, 1].set_yticks([])
    
    return axes
    
def plot_error(axes, dis_list, ex_list, ey_list, evx_list, evy_list, el_list, ew_list, ephi_list, color='red', label='20km/h',
        xlabel='Distance / (m)', xticks=[], yticks_xy=[], yticks_vxvy=[], yticks_lw=[], yticks_phi=[]):
    # 功能：绘制误差曲线
    
    # 绘制x0误差曲线
    axes[0, 0].plot(dis_list, ex_list, color, linestyle='-', linewidth=1, label=label)
    axes[0, 0].set_xlabel(xlabel, fontsize=10, labelpad=0)
    axes[0, 0].set_ylabel('Error / (m)', fontsize=10, labelpad=0)
    axes[0, 0].legend(loc='upper left', fontsize=8, ncol=1)
    axes[0, 0].set_title('Location error (X)', fontsize=10)
    axes[0, 0].set_xticks(xticks)
    axes[0, 0].set_yticks(yticks_xy)
    axes[0, 0].set_ylim(yticks_xy[0], yticks_xy[-1])
    
    # 绘制y0误差曲线
    axes[0, 1].plot(dis_list, ey_list, color, linestyle='-', linewidth=1, label=label)
    axes[0, 1].set_xlabel(xlabel, fontsize=10, labelpad=0)
    axes[0, 1].set_ylabel('Error / (m)', fontsize=10, labelpad=0)
    axes[0, 1].legend(loc='upper left', fontsize=8, ncol=1)
    axes[0, 1].set_title('Location error (Y)', fontsize=10)
    axes[0, 1].set_xticks(xticks)
    axes[0, 1].set_yticks(yticks_xy)
    axes[0, 1].set_ylim(yticks_xy[0], yticks_xy[-1])
    
    # 绘制vx误差曲线
    axes[1, 0].plot(dis_list, evx_list, color, linestyle='-', linewidth=1, label=label)
    axes[1, 0].set_xlabel(xlabel, fontsize=10, labelpad=0)
    axes[1, 0].set_ylabel('Error / (m/s)', fontsize=10, labelpad=0)
    axes[1, 0].legend(loc='upper left', fontsize=8, ncol=1)
    axes[1, 0].set_title('Velocity error (X)', fontsize=10)
    axes[1, 0].set_xticks(xticks)
    axes[1, 0].set_yticks(yticks_vxvy)
    axes[1, 0].set_ylim(yticks_vxvy[0], yticks_vxvy[-1])
    
    # 绘制vy误差曲线
    axes[1, 1].plot(dis_list, evy_list, color, linestyle='-', linewidth=1, label=label)
    axes[1, 1].set_xlabel(xlabel, fontsize=10, labelpad=0)
    axes[1, 1].set_ylabel('Error / (m/s)', fontsize=10, labelpad=0)
    axes[1, 1].legend(loc='upper left', fontsize=8, ncol=1)
    axes[1, 1].set_title('Velocity error (Y)', fontsize=10)
    axes[1, 1].set_xticks(xticks)
    axes[1, 1].set_yticks(yticks_vxvy)
    axes[1, 1].set_ylim(yticks_vxvy[0], yticks_vxvy[-1])
    
    # 绘制l误差曲线
    axes[2, 0].plot(dis_list, el_list, color, linestyle='-', linewidth=1, label=label)
    axes[2, 0].set_xlabel(xlabel, fontsize=10, labelpad=0)
    axes[2, 0].set_ylabel('Error / (m)', fontsize=10, labelpad=0)
    axes[2, 0].legend(loc='upper left', fontsize=8, ncol=1)
    axes[2, 0].set_title("Length error", fontsize=10)
    axes[2, 0].set_xticks(xticks)
    axes[2, 0].set_yticks(yticks_lw)
    axes[2, 0].set_ylim(yticks_lw[0], yticks_lw[-1])
    
    # 绘制w误差曲线
    axes[2, 1].plot(dis_list, ew_list, color, linestyle='-', linewidth=1, label=label)
    axes[2, 1].set_xlabel(xlabel, fontsize=10, labelpad=0)
    axes[2, 1].set_ylabel('Error / (m)', fontsize=10, labelpad=0)
    axes[2, 1].legend(loc='upper left', fontsize=8, ncol=1)
    axes[2, 1].set_title("Width error", fontsize=10)
    axes[2, 1].set_xticks(xticks)
    axes[2, 1].set_yticks(yticks_lw)
    axes[2, 1].set_ylim(yticks_lw[0], yticks_lw[-1])
    
    # 绘制phi误差曲线
    axes[3, 0].plot(dis_list, ephi_list, color, linestyle='-', linewidth=1, label=label)
    axes[3, 0].set_xlabel(xlabel, fontsize=10, labelpad=0)
    axes[3, 0].set_ylabel('Error / (rad)', fontsize=10, labelpad=0)
    axes[3, 0].legend(loc='upper left', fontsize=8, ncol=1)
    axes[3, 0].set_title("Azimuth error", fontsize=10)
    axes[3, 0].set_xticks(xticks)
    axes[3, 0].set_yticks(yticks_phi)
    axes[3, 0].set_ylim(yticks_phi[0], yticks_phi[-1])
    
    axes[3, 1].spines['right'].set_color('none')
    axes[3, 1].spines['left'].set_color('none')
    axes[3, 1].spines['top'].set_color('none')
    axes[3, 1].spines['bottom'].set_color('none')
    axes[3, 1].set_xticks([])
    axes[3, 1].set_yticks([])
    
    return axes
    
XG1, YG1 = 671574.879, 3529046.701
XG2, YG2 = 671566.978, 3529141.928

LENGTH = 4.925
WIDTH = 1.864
    
if __name__ == '__main__':
    # X Direction ######################################################
    # 读入GPS数据
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/GPS_data/line1_15.txt'
    time_gps_list_15, xg_list_15, yg_list_15, vxg_list_15, vyg_list_15, phig_list_15 = read_gps_data(
        filename, XG1, YG1, XG2, YG2, print_mode=True)
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/GPS_data/line1_40.txt'
    time_gps_list_40, xg_list_40, yg_list_40, vxg_list_40, vyg_list_40, phig_list_40 = read_gps_data(
        filename, XG1, YG1, XG2, YG2, print_mode=True)
    
    # 读入结果数据
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/2021-03-24-14-53-53/result.txt'
    time_result_list_15, frame_result_list_15, id_list_15, x_list_15, y_list_15, vx_list_15, vy_list_15, \
        x0_list_15, y0_list_15, z0_list_15, l_list_15, w_list_15, h_list_15, phi_list_15 = \
        read_result_data(filename, print_mode=True)
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/2021-03-24-15-10-36/result.txt'
    time_result_list_40, frame_result_list_40, id_list_40, x_list_40, y_list_40, vx_list_40, vy_list_40, \
        x0_list_40, y0_list_40, z0_list_40, l_list_40, w_list_40, h_list_40, phi_list_40 = \
        read_result_data(filename, print_mode=True)
    
    # 补偿系统时间差
    for i in range(len(time_result_list_15)):
        time_result_list_15[i] += 0.3
        x0_list_15[i] += 0.8
        y0_list_15[i] += 0.2
    for i in range(len(time_result_list_40)):
        time_result_list_40[i] += 0.3
        x0_list_40[i] += 0.8
        y0_list_40[i] += 0.2
    
    # 计算误差
    ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15 = compute_error(
        time_gps_list_15, time_result_list_15, xg_list_15, yg_list_15, vxg_list_15, vyg_list_15, LENGTH, WIDTH, phig_list_15,
            x0_list_15, y0_list_15, vx_list_15, vy_list_15, l_list_15, w_list_15, phi_list_15)
    ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40 = compute_error(
        time_gps_list_40, time_result_list_40, xg_list_40, yg_list_40, vxg_list_40, vyg_list_40, LENGTH, WIDTH, phig_list_40,
            x0_list_40, y0_list_40, vx_list_40, vy_list_40, l_list_40, w_list_40, phi_list_40)
    
    # 绘制15km/h工况的GPS数据和结果数据
    fig, axes = plt.subplots(4, 2, figsize=(5, 8))
    fig.canvas.set_window_title('X direction 15km/h')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.8)
    axes = plot_data(
        axes, time_gps_list_15, time_result_list_15, xg_list_15, yg_list_15, vxg_list_15, vyg_list_15, LENGTH, WIDTH, phig_list_15,
            x0_list_15, y0_list_15, vx_list_15, vy_list_15, l_list_15, w_list_15, phi_list_15, color='red')
    fig.savefig('X direction original data in 15.png', dpi=100)
    
    # 绘制40km/h工况的GPS数据和结果数据
    fig, axes = plt.subplots(4, 2, figsize=(5, 8))
    fig.canvas.set_window_title('X direction 40km/h')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.8)
    axes = plot_data(
        axes, time_gps_list_40, time_result_list_40, xg_list_40, yg_list_40, vxg_list_40, vyg_list_40, LENGTH, WIDTH, phig_list_40,
            x0_list_40, y0_list_40, vx_list_40, vy_list_40, l_list_40, w_list_40, phi_list_40, color='blue')
    fig.savefig('X direction original data in 40.png', dpi=100)
    
    # 截取有效数据
    x0_list_15, ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15 = crop_data(
        x0_list_15, ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15, min_dis=7, max_dis=43)
    x0_list_40, ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40 = crop_data(
        x0_list_40, ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40, min_dis=7, max_dis=43)
    
    # 绘制误差数据
    fig, axes = plt.subplots(4, 2, figsize=(5, 8))
    fig.canvas.set_window_title('X direction Error')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.8)
    axes = plot_error(
        axes, x0_list_15, ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15, color='red', label='20km/h',
            xlabel='X coordinate', xticks=[10, 20, 30, 40], yticks_xy=[0, 2], yticks_vxvy=[0, 2], yticks_lw=[0, 4], yticks_phi=[0, 2])
    axes = plot_error(
        axes, x0_list_40, ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40, color='blue', label='40km/h',
            xlabel='X coordinate', xticks=[10, 20, 30, 40], yticks_xy=[0, 2], yticks_vxvy=[0, 2], yticks_lw=[0, 4], yticks_phi=[0, 2])
    fig.savefig('X direction error.png', dpi=100)
    ####################################################################
    
    # Y Direction ######################################################
    # 读入GPS数据
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/GPS_data/line3_15.txt'
    time_gps_list_15, xg_list_15, yg_list_15, vxg_list_15, vyg_list_15, phig_list_15 = read_gps_data(
        filename, XG1, YG1, XG2, YG2, print_mode=True)
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/GPS_data/line3_40.txt'
    time_gps_list_40, xg_list_40, yg_list_40, vxg_list_40, vyg_list_40, phig_list_40 = read_gps_data(
        filename, XG1, YG1, XG2, YG2, print_mode=True)
    
    # 读入结果数据
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/2021-03-24-15-05-04/result.txt'
    time_result_list_15, frame_result_list_15, id_list_15, x_list_15, y_list_15, vx_list_15, vy_list_15, \
        x0_list_15, y0_list_15, z0_list_15, l_list_15, w_list_15, h_list_15, phi_list_15 = \
        read_result_data(filename, print_mode=True)
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/single-target/2021-03-24-15-17-35/result.txt'
    time_result_list_40, frame_result_list_40, id_list_40, x_list_40, y_list_40, vx_list_40, vy_list_40, \
        x0_list_40, y0_list_40, z0_list_40, l_list_40, w_list_40, h_list_40, phi_list_40 = \
        read_result_data(filename, print_mode=True)
    
    # 补偿系统时间差
    for i in range(len(time_result_list_15)):
        time_result_list_15[i] += 0.3
        x0_list_15[i] += 0.8
        y0_list_15[i] += 0.2
    for i in range(len(time_result_list_40)):
        time_result_list_40[i] += 0.3
        x0_list_40[i] += 0.8
        y0_list_40[i] += 0.2
    
    # 计算误差
    ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15 = compute_error(
        time_gps_list_15, time_result_list_15, xg_list_15, yg_list_15, vxg_list_15, vyg_list_15, LENGTH, WIDTH, phig_list_15,
            x0_list_15, y0_list_15, vx_list_15, vy_list_15, l_list_15, w_list_15, phi_list_15)
    ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40 = compute_error(
        time_gps_list_40, time_result_list_40, xg_list_40, yg_list_40, vxg_list_40, vyg_list_40, LENGTH, WIDTH, phig_list_40,
            x0_list_40, y0_list_40, vx_list_40, vy_list_40, l_list_40, w_list_40, phi_list_40)
    
    # 绘制15km/h工况的GPS数据和结果数据
    fig, axes = plt.subplots(4, 2, figsize=(5, 8))
    fig.canvas.set_window_title('Y direction 15km/h')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.8)
    axes = plot_data(
        axes, time_gps_list_15, time_result_list_15, xg_list_15, yg_list_15, vxg_list_15, vyg_list_15, LENGTH, WIDTH, phig_list_15,
            x0_list_15, y0_list_15, vx_list_15, vy_list_15, l_list_15, w_list_15, phi_list_15, color='red')
    fig.savefig('Y direction original data in 15.png', dpi=100)

    # 绘制40km/h工况的GPS数据和结果数据
    fig, axes = plt.subplots(4, 2, figsize=(5, 8))
    fig.canvas.set_window_title('Y direction 40km/h')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.8)
    axes = plot_data(
        axes, time_gps_list_40, time_result_list_40, xg_list_40, yg_list_40, vxg_list_40, vyg_list_40, LENGTH, WIDTH, phig_list_40,
            x0_list_40, y0_list_40, vx_list_40, vy_list_40, l_list_40, w_list_40, phi_list_40, color='blue')
    fig.savefig('Y direction original data in 40.png', dpi=100)
    
    # 截取有效数据
    y0_list_15, ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15 = crop_data(
        y0_list_15, ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15, min_dis=-8, max_dis=10)
    y0_list_40, ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40 = crop_data(
        y0_list_40, ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40, min_dis=-8, max_dis=10)
    
    # 绘制误差数据
    fig, axes = plt.subplots(4, 2, figsize=(5, 8))
    fig.canvas.set_window_title('Y direction Error')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.8)
    axes = plot_error(
        axes, y0_list_15, ex_list_15, ey_list_15, evx_list_15, evy_list_15, el_list_15, ew_list_15, ephi_list_15, color='red', label='20km/h',
            xlabel='Y coordinate', xticks=[-10, 0, 10], yticks_xy=[0, 2], yticks_vxvy=[0, 2], yticks_lw=[0, 2], yticks_phi=[0, 1])
    axes = plot_error(
        axes, y0_list_40, ex_list_40, ey_list_40, evx_list_40, evy_list_40, el_list_40, ew_list_40, ephi_list_40, color='blue', label='40km/h',
            xlabel='Y coordinate', xticks=[-10, 0, 10], yticks_xy=[0, 2], yticks_vxvy=[0, 2], yticks_lw=[0, 2], yticks_phi=[0, 1])
    fig.savefig('Y direction error.png', dpi=100)
    ####################################################################
    
    
    plt.show()
