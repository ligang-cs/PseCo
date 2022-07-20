import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import ipdb

name_list = ["[0,32]", "[32,64]", "[64,96]", "[96,128]", "[128,160]", "[160,192]", "[192,1e5]"]
save_root = "../checkpoints/"

def bar_figure_2_subplots(score_tp, score_fp):
   
    score_tp_list = score_tp.tolist()
    score_fp_list = score_fp.tolist()
    index = np.arange(len(name_list))
    
    plt.bar(index, score_tp_list, 0.3, tick_label=name_list, label="TPs", color='orange')
    plt.bar(index, score_fp_list, 0.3, tick_label=name_list, label="FPs", color='red')

    for a, b in zip(index, score_tp_list):
        plt.text(a, b, "%.2f"%b, ha='center', va='bottom', fontsize=12)

    for a, b in zip(index, score_fp_list):
        plt.text(a, b, "%.2f"%b, ha='center', va='bottom', fontsize=12)

    plt.title("Average score for scale ranges")
    plt.xlabel("scale range")
    plt.ylabel("score")
    plt.legend()
    plt.savefig(osp.join(save_root, "avg_score.png"))
    plt.close()


def bar_figure(values, title, xlabel, ylabel, file_name):
   
    value_list = values.tolist()
    index = np.arange(len(name_list))
    
    plt.bar(index, value_list, 0.3, tick_label=name_list)
    
    for a, b in zip(index, value_list):
        plt.text(a, b, "%.2f"%b, ha='center', va='bottom', fontsize=12)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(osp.join(save_root, file_name))
    plt.close()


def plot_distribution(tps_areas, fps_areas):
    score_range = np.linspace(0.3, 1, num=101, endpoint=True)
    
    for key, tps in tps_areas.items():
        
        index = []
        tps_num_list, fps_num_list, all_num_list = [], [], []

        tps = np.array(tps)
        fps = np.array(fps_areas[key])

        for i in range(len(score_range) - 1):
            tps_number = np.sum((tps < score_range[i+1]) & (tps > score_range[i]))
            fps_number = np.sum((fps < score_range[i+1]) & (fps > score_range[i]))
            all_num = tps_number + fps_number
            
            tps_num_list.append(tps_number)
            fps_num_list.append(fps_number)
            all_num_list.append(all_num)
            index.append((score_range[i] + score_range[i+1]) / 2)
        
        subfig = plt.subplot(2, 4, key+1)
        plt.plot(np.array(index), np.array(tps_num_list), label='TP', color='blue')
        plt.plot(np.array(index), np.array(fps_num_list), label='FP', color='red')
        plt.plot(np.array(index), np.array(all_num_list), label='All', color='orange')
        title = name_list[key]
        plt.title(title)
        plt.xlabel("cls score")
        plt.ylabel("Number")
        plt.legend()

    plt.show()
