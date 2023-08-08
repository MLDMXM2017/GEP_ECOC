# coding:utf-8
import os
import numpy as np
from glob import glob
from collections import Counter


def cal_confu_matrix(label, predict, class_num):
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)


def metrics(confu_mat_total, save_path="./"):
    '''
    :param confu_mat: 总的混淆矩阵
    backgound：是否干掉背景
    :return: txt写出混淆矩阵, precision，recall，IOU，f-score
    '''
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出precision，recall, f1-score，f1-m以及mIOU

    f1_m = []
    iou_m = []
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)

    f1_m = np.array(f1_m)
    iou_m = np.array(iou_m)
    if save_path is not None:
        with open(save_path + 'accuracy.txt', 'w') as f:
            f.write('OA:\t%.4f\n' % (oa * 100))
            f.write('kappa:\t%.4f\n' % (kappa * 100))
            f.write('mf1-score:\t%.4f\n' % (np.mean(f1_m) * 100))
            f.write('mIou:\t%.4f\n' % (np.mean(iou_m) * 100))

            # 写出precision
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / raw_sum[i]) * 100))
            f.write('\n')

            # 写出recall
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / col_sum[i]) * 100))
            f.write('\n')

            # 写出f1-score
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(f1_m[i]) * 100))
            f.write('\n')

            # 写出 IOU
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(iou_m[i]) * 100))
            f.write('\n')