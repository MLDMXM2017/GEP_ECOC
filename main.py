# coding:utf-8
import geppy as gep
from deap import creator, base, tools
import copy
import numpy as np
import time
import pandas as pd
import math
import os
import random
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import customize_tools as ct
from ecoc_tools import SimpleECOCClassifier, SimpleECOCClassifier2
from Classifiers.BaseClassifier2 import GBC
from Evaluate.Evaluation_tool import Evaluation
import sel_cx_mut
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息


# 定义适应度函数
def evaluate(individuals, estimator_type):
    # 获取生成树中的所有节点
    # print(individuals)   # individuals 长度为1
    for individual in individuals:
        # print(str(individual))  # 表达式类型
        agene = individual.kexpression
        gene = []
        for i in range(len(agene)):
            gene.append(agene[i].name)
        # print(gene)  # 列表类型
        index = np.unique(target).tolist()  # 记录原始数据中共多少个类别
        node_bank = ct.generate_tree(str(individual), total_list=total_list, input_names_list=input_names_list)
        # 读取节点中字符串的值，并组成ECOC编码矩阵
        estimator = gbc.get_base_clf(estimator_type)  # 选择SVM作为基分类器
        node_bank_de_duplication = ct.de_duplication_node_bank(node_bank)  # 去重
        node_bank_final = ct.transpose(node_bank_de_duplication)  # 转置，并将list类型的矩阵转化为ndarray类型的矩阵
        matrix = np.array(node_bank_final)
        Code_Matrix = matrix  # M是函数返回值即matrix（编码矩阵）
        f2 = open("./Logging/" + file_path + "/" + file_name + "迭代结果.txt", 'a+')
        print("局部优化前的原始编码矩阵：", file=f2)
        print(Code_Matrix, file=f2)
        print("", file=f2)
        f2.close()

        sec_M = SimpleECOCClassifier2(estimator, Code_Matrix)
        sec_M.fit(train_x_svmrfe30, train_y_svmrfe30, train_x_rf30, train_y_rf30, train_x_bsswss30, train_y_bsswss30, train_x_svmrfe50, train_y_svmrfe50, train_x_rf50, train_y_rf50, train_x_bsswss50, train_y_bsswss50, train_x_svmrfe70, train_y_svmrfe70, train_x_rf70, train_y_rf70, train_x_bsswss70, train_y_bsswss70)
        pred_label_M = sec_M.predict(validate_x_svmrfe30, validate_x_rf30, validate_x_bsswss30, validate_x_svmrfe50, validate_x_rf50, validate_x_bsswss50, validate_x_svmrfe70, validate_x_rf70, validate_x_bsswss70)
        result = Evaluation(validate_y, pred_label_M).evaluation(accuracy=True, precision=True, sensitivity=True, Fscore=True)
        accuracy = result['accuracy']
        Fscore = result['Fscore']
        # print("局部优化前的原始编码矩阵的Fscore值：", Fscore)
        # print("局部优化前的原始编码矩阵的Accuracy值：", accuracy)
        # 局部优化
        Fscore_new, Acc_new, node_bank_de_duplication_T, old_Per_Cla_Acc, new_Per_Cla_Acc, tough_classes_list = ct.local_optimization(validate_y, pred_label_M, node_bank_de_duplication, Fscore, accuracy, n_class, estimator_type, train_x_svmrfe30, train_y_svmrfe30, train_x_rf30, train_y_rf30, train_x_bsswss30, train_y_bsswss30, validate_x_svmrfe30, validate_x_rf30, validate_x_bsswss30, train_x_svmrfe50, train_y_svmrfe50, train_x_rf50, train_y_rf50, train_x_bsswss50, train_y_bsswss50, validate_x_svmrfe50, validate_x_rf50, validate_x_bsswss50, train_x_svmrfe70, train_y_svmrfe70, train_x_rf70, train_y_rf70, train_x_bsswss70, train_y_bsswss70, validate_x_svmrfe70, validate_x_rf70, validate_x_bsswss70, index, file_name, file_path)
        new_acc.append(Acc_new)
        new_value.append(Fscore_new)
        new_matrix.append(node_bank_de_duplication_T)
        new_toughclasses.append(tough_classes_list)
        old_matrix.append(node_bank_de_duplication)
        return Fscore,


def opt_calculate(node_bank_de_duplication, estimator_type, label, raw_node_bank_de_duplication, best_ind_step):
    # index = np.unique(label).tolist()  # 记录原始数据中共多少个类别
    estimator = gbc.get_base_clf(estimator_type)  # 选择SVM作为基分类器
    raw_node_bank_final = ct.transpose(raw_node_bank_de_duplication)  # 转置，并将list类型的矩阵转化为ndarray类型的矩阵
    raw_matrix = np.array(raw_node_bank_final)
    raw_Code_matrix = raw_matrix  # M是函数返回值即matrix（编码矩阵）
    raw_sec = SimpleECOCClassifier2(estimator, raw_Code_matrix)
    raw_sec.fit(train_x_svmrfe30, train_y_svmrfe30, train_x_rf30, train_y_rf30, train_x_bsswss30, train_y_bsswss30, train_x_svmrfe50, train_y_svmrfe50, train_x_rf50, train_y_rf50, train_x_bsswss50, train_y_bsswss50, train_x_svmrfe70, train_y_svmrfe70, train_x_rf70, train_y_rf70, train_x_bsswss70, train_y_bsswss70)
    raw_pred_label = raw_sec.predict(test_x_svmrfe30, test_x_rf30, test_x_bsswss30, test_x_svmrfe50, test_x_rf50, test_x_bsswss50, test_x_svmrfe70, test_x_rf70, test_x_bsswss70)
    result = Evaluation(test_y, raw_pred_label).evaluation(accuracy=True, precision=True, sensitivity=True, Fscore=True)
    raw_accuracy = result['accuracy']
    raw_Fscore = result['Fscore']
    print("原始编码矩阵为", file=f)
    print(raw_matrix, file=f)
    print("原始编码矩阵的Fscore值为：", raw_Fscore, file=f)
    print("原始编码矩阵的Accuracy值为", raw_accuracy, file=f)
    raw_percla_acc =[]
    raw_CMlist = confusion_matrix(test_y, raw_pred_label).tolist()
    for i in range(len(raw_CMlist)):
        total = sum(raw_CMlist[i])
        if total != 0:
            acc = raw_CMlist[i][i] / total
        else:
            acc = 0
        # print("当前矩阵的第", index[i], "类别的正确率为", acc)
        raw_percla_acc.append(acc)
    print("原始编码矩阵的分类准确率为", raw_percla_acc, file=f)
    print("", file=f)
    print("局部优化中间过程的难分类对包括：", best_ind_step, file=f)
    node_bank_final = ct.transpose(node_bank_de_duplication)  # 转置，并将list类型的矩阵转化为ndarray类型的矩阵
    matrix = np.array(node_bank_final)
    Code_matrix = matrix  # M是函数返回值即matrix（编码矩阵）
    print("", file=f)
    # 根据编码矩阵对数据进行训练和预测，得到F1或Acc
    sec = SimpleECOCClassifier2(estimator, Code_matrix)
    sec.fit(train_x_svmrfe30, train_y_svmrfe30, train_x_rf30, train_y_rf30, train_x_bsswss30, train_y_bsswss30, train_x_svmrfe50, train_y_svmrfe50, train_x_rf50, train_y_rf50, train_x_bsswss50, train_y_bsswss50, train_x_svmrfe70, train_y_svmrfe70, train_x_rf70, train_y_rf70, train_x_bsswss70, train_y_bsswss70)
    pred_label = sec.predict(test_x_svmrfe30, test_x_rf30, test_x_bsswss30, test_x_svmrfe50, test_x_rf50, test_x_bsswss50, test_x_svmrfe70, test_x_rf70, test_x_bsswss70)
    result = Evaluation(test_y, pred_label).evaluation(accuracy=True, precision=True, sensitivity=True, Fscore=True)
    accuracy = result['accuracy']
    Fscore = result['Fscore']
    print("局部优化后的编码矩阵为", file=f)
    print(Code_matrix, file=f)
    print("局部优化后的编码矩阵的Fscore值为：", Fscore, file=f)
    print("局部优化后的编码矩阵的Accuracy值为", accuracy, file=f)
    CMlist = confusion_matrix(test_y, pred_label).tolist()
    for i in range(len(CMlist)):
        total = sum(CMlist[i])
        if total != 0:
            acc = CMlist[i][i] / total + 0.000000001
        else:
            acc = 0
        # print("当前矩阵的第", index[i], "类别的正确率为", acc)
        new_percla_acc.append(acc)
    print("局部优化后的编码矩阵的分类准确率为", new_percla_acc, file=f)
    print("", file=f)
    print("")
    print("本次实验的运行结果", Fscore, file=f)
    print("*****************")
    print("", file=f)
    # 返回评价值（目标是使其最大化）
    return Fscore, accuracy, raw_Fscore, raw_accuracy


if __name__ == '__main__':
    # 启动进化
    n_pop = 80  # 种群个体数
    n_gen = 40  # 迭代数
    datafile = ["balance"]
    estimators_type = ["SVM"]  # ["KNN", "Bayes", "DTree", "Logi", "SVM"]
    random_states = random.sample(range(1, 10000), 10)  # 生成随机种子
    for index in range(len(datafile)):
        if datafile[index] == "abalone" or datafile[index] == "cmc" or datafile[index] == "iris" or datafile[index] =="wine" \
                or datafile[index] == "vertebral" or datafile[index] == "waveform-+noise" or datafile[index] == "waveform" \
                or datafile[index] == "balance" or datafile[index] == "Leukemia1" or datafile[index] == "Leukemia2"or datafile[index] == "Lung1":
            n_class = 3
        elif datafile[index] == "zoo" or datafile[index] == "DLBCL" or datafile[index] == "Lung2" or datafile[index] == "SRBCT":
            n_class = 4
        elif datafile[index] == "glass" or datafile[index] == "ecoli" or datafile[index] == "winequality-red"or datafile[index] == "heart-disease.hungarian"or datafile[index] == "Breast":
            n_class = 5
        elif datafile[index] == "winequality-white" or datafile[index] == "dermatology":
            n_class = 6
        elif datafile[index] == "Cancers":
            n_class = 8
        elif datafile[index] == "yeast":
            n_class = 9
        else:
            n_class = 14
        depth_max = math.floor(np.log2(2 * n_class))  # 完全二叉树的最大深度
        seed_num = 3 ** n_class - 1  # 个体数
        for estimator_type in estimators_type:
            exp_ten_value = []
            final_fscore = []
            final_acc = []
            final_raw_fscore = []
            final_raw_acc = []
            start = time.time()
            if not os.path.exists("./Logging/" + datafile[index] + "_" + estimator_type):
                os.mkdir("./Logging/" + datafile[index] + "_" + estimator_type)
            f = open("./Logging/" + datafile[index] + "_" + estimator_type + "/最终结果.txt", 'w+')
            for random_state_index in range(len(random_states)):
                file_path = datafile[index] + "_" + estimator_type
                file_name = "第" + str(random_state_index+1) + "次实验_"
                print("*****************", file=f)
                print("第", random_state_index+1, "次实验", file=f)
                # 读取数据
                data = pd.read_csv("./Dataset/" + datafile[index] + ".csv", encoding='utf-8')
                # data.drop([data.columns[0]], axis=1, inplace=True)  # 删除第一列（序号列）
                target = data[data.columns[data.shape[1] - 1]]  # 获取最后一列（标签列）
                data.drop([data.columns[data.shape[1] - 1]], axis=1, inplace=True)  # 删除最后一列
                data = data.values
                target = target.values
                mm = MinMaxScaler()  # 归一化
                data = mm.fit_transform(data)
                train_X, train_x, train_Y, train_y = train_test_split(data, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x, validate_x, test_y, validate_y = train_test_split(train_X, train_Y, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                gbc = GBC(train_x, train_y)
                new_value = []
                new_matrix = []
                new_acc = []
                new_percla_acc = []
                new_toughclasses = []
                old_matrix = []
                data_svmrfe30 = ct.process_select_svmrfe30(data, target)
                data_rf30 = ct.process_select_forest30(data, target)
                data_bsswss30 = ct.process_select_bsswss30(data, target)

                data_svmrfe50 = ct.process_select_svmrfe50(data, target)
                data_rf50 = ct.process_select_forest50(data, target)
                data_bsswss50 = ct.process_select_bsswss50(data, target)

                data_svmrfe70 = ct.process_select_svmrfe70(data, target)
                data_rf70 = ct.process_select_forest70(data, target)
                data_bsswss70 = ct.process_select_bsswss70(data, target)

                train_X_svmrfe30, train_x_svmrfe30, train_Y_svmrfe30, train_y_svmrfe30 = train_test_split(data_svmrfe30, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_svmrfe30, validate_x_svmrfe30, test_y_svmrfe30, validate_y_svmrfe30 = train_test_split(train_X_svmrfe30, train_Y_svmrfe30, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                train_X_rf30, train_x_rf30, train_Y_rf30, train_y_rf30 = train_test_split(data_rf30, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_rf30, validate_x_rf30, test_y_rf30, validate_y_rf30 = train_test_split(train_X_rf30, train_Y_rf30, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                train_X_bsswss30, train_x_bsswss30, train_Y_bsswss30, train_y_bsswss30 = train_test_split(data_bsswss30, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_bsswss30, validate_x_bsswss30, test_y_bsswss30, validate_y_bsswss30 = train_test_split(train_X_bsswss30, train_Y_bsswss30, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)

                train_X_svmrfe50, train_x_svmrfe50, train_Y_svmrfe50, train_y_svmrfe50 = train_test_split(data_svmrfe50, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_svmrfe50, validate_x_svmrfe50, test_y_svmrfe50, validate_y_svmrfe50 = train_test_split(train_X_svmrfe50, train_Y_svmrfe50, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                train_X_rf50, train_x_rf50, train_Y_rf50, train_y_rf50 = train_test_split(data_rf50, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_rf50, validate_x_rf50, test_y_rf50, validate_y_rf50 = train_test_split(train_X_rf50, train_Y_rf50, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                train_X_bsswss50, train_x_bsswss50, train_Y_bsswss50, train_y_bsswss50 = train_test_split(data_bsswss50, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_bsswss50, validate_x_bsswss50, test_y_bsswss50, validate_y_bsswss50 = train_test_split(train_X_bsswss50, train_Y_bsswss50, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)

                train_X_svmrfe70, train_x_svmrfe70, train_Y_svmrfe70, train_y_svmrfe70 = train_test_split(data_svmrfe70, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_svmrfe70, validate_x_svmrfe70, test_y_svmrfe70, validate_y_svmrfe70 = train_test_split(train_X_svmrfe70, train_Y_svmrfe70, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                train_X_rf70, train_x_rf70, train_Y_rf70, train_y_rf70 = train_test_split(data_rf70, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_rf70, validate_x_rf70, test_y_rf70, validate_y_rf70 = train_test_split(train_X_rf70, train_Y_rf70, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                train_X_bsswss70, train_x_bsswss70, train_Y_bsswss70, train_y_bsswss70 = train_test_split(data_bsswss70, target, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)
                test_x_bsswss70, validate_x_bsswss70, test_y_bsswss70, validate_y_bsswss70 = train_test_split(train_X_bsswss70, train_Y_bsswss70, test_size=0.5, random_state=random_states[random_state_index], shuffle=True)

                # 输入四个叶子节点的值
                total_list = ct.select_input(n_class, seed_num, depth_max)
                alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                input_names_list = alphabet[:len(total_list)]
                # 创建原始集
                pset = gep.PrimitiveSet('Main', input_names=input_names_list)
                pset.add_function(ct.ternary_add, 2)
                pset.add_function(ct.ternary_sub, 2)
                pset.add_function(ct.ternary_mul, 2)
                pset.add_function(ct.ternary_and, 2)
                pset.add_function(ct.ternary_or, 2)
                # pset.add_function(ct.ternary_reverse, 2)
                pset.add_function(ct.ternary_oddeven, 2)
                pset.add_function(ct.ternary_halfhalf, 2)
                # 创建个体和种群
                creator.create("FitnessMax", base.Fitness, weights=(1,))
                creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

                # 注册个体和种群创建算子
                h = 7  # 基因头长度为7
                n_genes = 1  # 染色体中共两个基因
                toolbox = gep.Toolbox()
                toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
                toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes)
                # toolbox.register('population', tools.initRepeat, list, toolbox.individual)
                toolbox.register('population', ct.initRepeat, list, toolbox.individual, total_list=total_list, input_names_list=input_names_list, n_class=n_class)
                toolbox.register('compile', gep.compile_, pset=pset,)
                toolbox.register('evaluate', evaluate, estimator_type=estimator_type)
                # 死亡惩罚，当个体不满足约束条件时，设置其个体适应度为-100
                # toolbox.decorate('evaluate', tools.DeltaPenalty(feasible, -100))
                # 注册遗传算子
                # 注册选择算子
                toolbox.register('select', tools.selStochasticUniversalSampling)
                # 注册变异算子和交叉算子
                toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=2 / (2 * h + 1), pb=0.1)
                toolbox.register('mut_invert', gep.invert, pb=0.1)
                toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
                toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
                toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
                toolbox.register('cx_1p', gep.crossover_one_point, pb=0.1)
                toolbox.register('cx_2p', gep.crossover_two_point, pb=0.1)
                toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
                # toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')  # 1p: 个体中预期的单点突变
                # toolbox.pbs['mut_ephemeral'] = 1  # 我们也可以通过 pbs 属性给出概率

                # 统计 监控
                stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
                stats.register("avg", np.mean)
                stats.register("std", np.std)
                stats.register("min", np.min)
                stats.register("max", np.max)



                pop = toolbox.population(n=n_pop)
                pop_copy = copy.deepcopy(pop)
                hof = tools.HallOfFame(3)  # only record the best three individuals ever found in all generations

                pop2, log = sel_cx_mut.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=8, stats=stats, hall_of_fame=hof, verbose=True, total_list=total_list, input_names_list=input_names_list, n_class=n_class, file_name=file_name, file_path=file_path)
                all_index = np.unique(target).tolist()  # 记录原始数据中共多少个类别
                print("该数据共有以下类别label：", all_index, file=f)
                print("", file=f)
                # pop2是经过选择、交叉、变异和精英保留后的种群
                print("最佳个体：", hof[0], file=f)
                print("最佳个体的基因型:", file=f)
                for gene in hof[0]:
                    print(str(gene.kexpression), file=f)
                # print(type(hof[0]))
                best_ind = hof[0]
                # extract statistics:
                maxFitnessValues, meanFitnessValues = log.select("max", "avg")
                best_opt_col_list = new_matrix[new_value.index(max(new_value))]
                best_ind_step = new_toughclasses[new_value.index(max(new_value))]
                raw_matrix = old_matrix[new_value.index(max(new_value))]
                best_opt_value, best_opt_acc, best_raw_Fscore, best_raw_accuracy = opt_calculate(best_opt_col_list, estimator_type, target, raw_matrix, best_ind_step)
                # 绘制迭代图:
                sns.set_style("whitegrid")
                plt.plot(maxFitnessValues, color='red')
                plt.plot(meanFitnessValues, color='green')
                plt.xlabel('Generation')
                plt.ylabel('Max / Average Fitness')
                plt.title('Min and Average Fitness over Generations')
                dir = "./Iteration_Chart/" + datafile[index] + "_" + estimator_type
                dir2 = dir + "/第" + str(random_state_index + 1) + "次" + ".jpg"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                plt.savefig(dir2)
                plt.show()
                # final_fscore.append(calculate(best_ind, estimator_type))
                final_fscore.append(best_opt_value)
                final_acc.append(best_opt_acc)
                final_raw_fscore.append(best_raw_Fscore)
                final_raw_acc.append(best_raw_accuracy)
            avg_final_fscore = sum(final_fscore)/len(final_fscore)
            best_final_fscore = max(final_fscore)
            avg_final_acc = sum(final_acc)/len(final_acc)
            best_final_acc = max(final_acc)
            avg_final_raw_fscore = sum(final_raw_fscore)/len(final_raw_fscore)
            best_final_raw_fscore = max(final_raw_fscore)
            avg_final_raw_acc = sum(final_raw_acc)/len(final_raw_acc)
            best_final_raw_acc = max(final_raw_acc)

            print("*********************", file=f)
            print("十次实验的平均Acc为（局部优化前）：", avg_final_raw_acc, file=f)
            print("十次实验的最优Acc为（局部优化前）：", best_final_raw_acc, file=f)

            print("十次实验的平均Fscore为（局部优化前）：", avg_final_raw_fscore, file=f)
            print("十次实验的最优Fscore为（局部优化前）：", best_final_raw_fscore, file=f)

            print("*********************", file=f)
            print("十次实验的平均Acc为（局部优化后）：", avg_final_acc, file=f)
            print("十次实验的最优Acc为（局部优化后）：", best_final_acc, file=f)
            print("", file=f)
            print("十次实验的平均Fscore为（局部优化后）：", avg_final_fscore, file=f)
            print("十次实验的最优Fscore为（局部优化后）：", best_final_fscore, file=f)
            print("*********************", file=f)
            # symplified_best = gep.simplify(best_ind)
            # print('Symplified best individual: ')
            # print(symplified_best)
            end = time.time()
            print("运行时长：", end - start, file=f)
            print()
            print("", file=f)
