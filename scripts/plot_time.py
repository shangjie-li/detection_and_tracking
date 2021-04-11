# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

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
        
        print('frame:%d amount:%d segmentation:%f projection:%f fusion:%f tracking:%f display:%f all:%f' % (
            frame, amount, time_segmentation, time_projection, time_fusion, time_tracking, time_display, time_all))
    
    # 设置画布
    fig, axes = plt.subplots(2, 1, figsize=(4.5, 6))
    fig.canvas.set_window_title('Time cost')
    plt.subplots_adjust(left=0.125, right=0.875, bottom=0.1, top=0.9, wspace=0.0, hspace=0.5)
    
    # 绘制amount
    axes[0].plot(frame_list, amount_list, color='red', linestyle='-', linewidth=1, label='amount      ')
    axes[0].set_ylim(-0.5, 13)
    axes[0].legend(loc='upper right', fontsize=8, ncol=1)
    axes[0].set_title('Number of objects - Frame', fontsize=10)
    
    # 绘制time_cost
    axes[1].plot(frame_list, time_segmentation_list, color='blue', linestyle='-', linewidth=1, label='segmentation')
    axes[1].plot(frame_list, time_projection_list, color='purple', linestyle='-', linewidth=1, label='projection  ')
    axes[1].plot(frame_list, time_fusion_list, color='green', linestyle='-', linewidth=1, label='fusion      ')
    axes[1].plot(frame_list, time_tracking_list, color='orange', linestyle='-', linewidth=1, label='tracking    ')
    axes[1].plot(frame_list, time_all_list, color='red', linestyle='-', linewidth=1, label='all         ')
    axes[1].set_yticks([0, 0.05, 0.10])
    axes[1].set_ylim(-0.005, 0.13)
    axes[1].legend(loc='upper center', fontsize=8, ncol=3)
    axes[1].set_title('Time cost - Frame', fontsize=10)
    
    fig.savefig('amount and time cost.png', dpi=100)
    plt.show()
    
    