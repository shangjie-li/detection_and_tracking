# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 读入结果文件
    filename = '/home/lishangjie/detection-and-tracking_doc/result/2021-04-07-experiment/multiple-target/2021-03-23-16-15-33/time_cost.txt'
    with open(filename) as fob:
        infos = fob.readlines()
    print(type(infos), len(infos))
    print()
    
    # 初始化结果列表
    frame_list = []
    amount_list = []
    time_segmentation_list = []
    time_projection_list = []
    time_fusion_list = []
    time_tracking_list = []
    time_display_list = []
    time_all_list = []
    
    # 统计不同目标数量(0 ~ num - 1)对应的平均耗时，frequencys等列表的索引为目标数量
    num = 20
    frequencys = [0] * num
    costs_segmentation = [0] * num
    costs_projection = [0] * num
    costs_fusion = [0] * num
    costs_tracking = [0] * num
    costs_display = [0] * num
    costs_all = [0] * num
    
    # 逐行读取
    num_frame = len(infos)
    for i in range(1, num_frame):
        info = list(infos[i])
        
        # 读取frame
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        frame = int(temp_str)
        frame_list.append(frame)
        
        # 读取amount
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        amount = int(temp_str)
        amount_list.append(amount)
        
        # 读取time_segmentation
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time_segmentation = float(temp_str)
        time_segmentation_list.append(time_segmentation)
        
        # 读取time_projection
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time_projection = float(temp_str)
        time_projection_list.append(time_projection)
        
        # 读取time_fusion
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time_fusion = float(temp_str)
        time_fusion_list.append(time_fusion)
        
        # 读取time_tracking
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time_tracking = float(temp_str)
        time_tracking_list.append(time_tracking)
        
        # 读取time_display
        while info.pop(0) != ':': pass
        k = 0
        while info[k] != ' ': k += 1
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time_display = float(temp_str)
        time_display_list.append(time_display)
        
        # 读取time_all
        while info.pop(0) != ':': pass
        k = len(info)
        temp_str = ''
        for j in range(k): temp_str += info[j]
        time_all = float(temp_str)
        time_all_list.append(time_all)
        
        # 统计不同目标数量对应的计算用时
        frequencys[amount] += 1
        costs_segmentation[amount] += time_segmentation
        costs_projection[amount] += time_projection
        costs_fusion[amount] += time_fusion
        costs_tracking[amount] += time_tracking
        costs_display[amount] += time_display
        costs_all[amount] += time_all
        
        print('frame:%d amount:%d segmentation:%f projection:%f fusion:%f tracking:%f display:%f all:%f' % (
            frame, amount, time_segmentation, time_projection, time_fusion, time_tracking, time_display, time_all))
    
    # 计算平均值
    for j in range(len(frequencys)):
        if frequencys[j] != 0:
            costs_segmentation[j] /= frequencys[j]
            costs_projection[j] /= frequencys[j]
            costs_fusion[j] /= frequencys[j]
            costs_tracking[j] /= frequencys[j]
            costs_display[j] /= frequencys[j]
            costs_all[j] /= frequencys[j]
    
    amounts = []
    costs_segmentation_a = []
    costs_projection_a = []
    costs_fusion_a = []
    costs_tracking_a = []
    costs_display_a = []
    costs_all_a = []
    for j in range(len(frequencys)):
        if frequencys[j] != 0 and j >= 4:
            amounts.append(j)
            costs_segmentation_a.append(costs_segmentation[j])
            costs_projection_a.append(costs_projection[j])
            costs_fusion_a.append(costs_fusion[j])
            costs_tracking_a.append(costs_tracking[j])
            costs_display_a.append(costs_display[j])
            costs_all_a.append(costs_all[j])
    
    # 设置画布
    fig, axes = plt.subplots(3, 1, figsize=(5, 8))
    fig.canvas.set_window_title('Time cost')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    
    # 绘制amount
    axes[0].plot(frame_list, amount_list, color='deeppink', linestyle='-', linewidth=1.3, label='amount')
    axes[0].tick_params(direction='in')
    axes[0].set_xticks([0, 50, 100, 150, 200, 250])
    axes[0].set_yticks([2, 4, 6, 8, 10, 12])
    axes[0].set_xlim(0, 280)
    axes[0].set_ylim(2, 12)
    axes[0].spines['right'].set_color('none')
    axes[0].spines['top'].set_color('none')
    axes[0].set_title('Number of objects - Frame', fontsize=10)
    
    # 绘制time_cost
    axes[1].plot(frame_list, time_segmentation_list, color='darkorange', linestyle='-', linewidth=1.3, label='segment')
    axes[1].plot(frame_list, time_projection_list, color='slategray', linestyle='-', linewidth=1.3, label='project')
    axes[1].plot(frame_list, time_fusion_list, color='cornflowerblue', linestyle='-', linewidth=1.3, label='fuse   ')
    axes[1].plot(frame_list, time_tracking_list, color='mediumseagreen', linestyle='-', linewidth=1.3, label='track  ')
    axes[1].plot(frame_list, time_all_list, color='deeppink', linestyle='-', linewidth=1.3, label='all    ')
    axes[1].tick_params(direction='in')
    axes[1].set_xticks([0, 50, 100, 150, 200, 250])
    axes[1].set_yticks([0, 0.05, 0.10])
    axes[1].set_xlim(0, 280)
    axes[1].set_ylim(-0.005, 0.10)
    axes[1].spines['right'].set_color('none')
    axes[1].spines['top'].set_color('none')
    axes[1].legend(loc='upper right', fontsize=8, ncol=3)
    axes[1].set_title('Time cost - Frame', fontsize=10)
    
    # 绘制目标数量与耗时之间的关系
    axes[2].bar(amounts, costs_all_a, width=0.35, color='deeppink', label='all')
    axes[2].tick_params(direction='in')
    axes[2].set_xticks([4, 5, 6, 7, 8, 9, 10])
    axes[2].set_yticks([0.05, 0.06, 0.07, 0.08])
    axes[2].set_xlim(3.5, 10.5)
    axes[2].set_ylim(0.05, 0.08)
    axes[2].spines['right'].set_color('none')
    axes[2].spines['top'].set_color('none')
    axes[2].tick_params(bottom=False, top=False, left=True, right=False)
    for x, y in zip(amounts, costs_all_a):
        axes[2].text(x - 0.285, y + 0.001, '%.3f' % y, fontsize=8)
    axes[2].set_title('Time cost - Number of objects', fontsize=10)
    
    fig.savefig('amount and time cost.png', dpi=100)
    
    plt.show()
    
    
