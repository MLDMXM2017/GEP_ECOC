# coding:utf-8
import random
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, recall_score
import copy
from Evaluate.Evaluation_tool import Evaluation
from ecoc_tools import SimpleECOCClassifier, SimpleECOCClassifier2
from sklearn.neighbors import NearestNeighbors
import collections
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from Classifiers.BaseClassifier import get_base_clf
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 生成种子库seed_bank，它是一个长度为seed_num的列表，列表中的元素为长度为n_class的列表seeds，seeds为一个长度为n_class的三进制字符串
def generate_ternary(n_class, seed_num):
    seed_bank = []  # 种子库
    i = 0
    # 保证种子库中不会产生重复的种子
    while i != seed_num:
        seeds = []
        for j in range(n_class):
            seed = random.randint(-1, 1)
            seeds.append(seed)
        if seeds not in seed_bank:
            seed_bank.append(seeds)
            i += 1
        else:
            i = i

    return seed_bank


# 输入序列的合法性检验
def select_input_legality_check(node_bank):
    flag = True
    # 列的要求
    # 限定编码矩阵的最小列数为4，限定编码矩阵的最大列数2 * n_class
    # if len(node_bank) < 4 or len(node_bank) > 2 * n_class:
    for node_new in node_bank:
        node = node_new[2:]
        if not (-1 in node and 1 in node):  # 每一列必须包含-1和1
            flag = False
        # if node == node[::-1]:  # 不能含有相反的列
        #     flag = False
    # matrix = np.array(node_bank2)
    return flag


def select_input_legality_check_new(node_bank):
    flag = True
    node_bank_temp = []
    node_bank_temp2 = []
    for node_new in node_bank:
        node = node_new[2:]
        node2 = node_new[2:]
        if not (-1 in node and 1 in node):  # 每一列必须包含-1和1
            flag = False
        neg_node = [-x for x in node]
        if node not in node_bank_temp and neg_node not in node_bank_temp:
            node_bank_temp.append(node)
        pos_node = [x for x in node2]
        if node2 not in node_bank_temp2 and pos_node not in node_bank_temp2:
            node_bank_temp2.append(node2)
    if len(node_bank) != len(node_bank_temp):
        flag = False
    if len(node_bank) != len(node_bank_temp2):
        flag = False
    return flag


# 从种子库中随机选择2**depth_max-1个序列作为输入（供1个可供选择的输入组合（A,B,C,D））
def select_input(n_class, seed_num, depth_max):
    total_list = []
    seed_bank = generate_ternary(n_class+2, seed_num)  # 生成种子库
    inputs_num = 2**(depth_max-1)  # 确定叶子结点数
    flag = True
    total = []
    while flag:
        total_index = []
        for i in range(inputs_num):
            list_index_num = int(random.gauss(mu=seed_num / 2, sigma=seed_num / inputs_num))
            while list_index_num in total_index or list_index_num < 0 or list_index_num >= seed_num:
                list_index_num = int(random.gauss(mu=seed_num / 2, sigma=seed_num / inputs_num))
            total_index.append(list_index_num)
            total.append(seed_bank[list_index_num])
        # total = random.sample(seed_bank, inputs_num)  # 从种子库中随机寻找inputs_num数量的叶子结点
        if select_input_legality_check(total):
            flag = False
        else:
            total = []
    print("total", total)
    print("输入种子为")
    for p in total:
        print([p])
        total_list.append([p])
    return total_list


# 生成运算符
def ternary_add(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    carry_now = 0  # 当前位的值
    carry_next = 0  # 进位的值
    for i, j in zip(x1, x2):
        if i + j + carry_next < -2:
            carry_now = 0
            carry_next = -1
            seeds.append(carry_now)
        elif i + j + carry_next == -2:
            carry_now = 1
            carry_next = -1
            seeds.append(carry_now)
        elif -2 < i + j + carry_next < 2:
            carry_now = i + j + carry_next
            carry_next = 0
            seeds.append(carry_now)
        elif i + j + carry_next == 2:
            carry_now = -1
            carry_next = 1
            seeds.append(carry_now)
        elif i + j + carry_next > 2:
            carry_now = 0
            carry_next = 1
            seeds.append(carry_now)
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 add x2', x1, 'add', x2, '=', seeds)
    return seeds
def ternary_sub(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    carry_now = 0  # 当前位的值
    carry_next = 0  # 进位的值
    for i, j in zip(x1, x2):
        if i - j + carry_next < -2:
            carry_now = 0
            carry_next = -1
            seeds.append(carry_now)
        elif i - j + carry_next == -2:
            carry_now = 1
            carry_next = -1
            seeds.append(carry_now)
        elif -2 < i - j + carry_next < 2:
            carry_now = i - j + carry_next
            carry_next = 0
            seeds.append(carry_now)
        elif i - j + carry_next == 2:
            carry_now = -1
            carry_next = 1
            seeds.append(carry_now)
        elif i - j + carry_next > 2:
            carry_now = 0
            carry_next = 1
            seeds.append(carry_now)
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 sub x2', x1, 'sub', x2, '=', seeds)
    return seeds
def ternary_mul(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    for i, j in zip(x1, x2):
        seeds.append(i * j)
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 mul x2', x1, 'mul', x2, '=', seeds)
    return seeds
def ternary_and(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    for i, j in zip(x1, x2):
        if i == 0 or j == 0:
            seeds.append(0)
        elif i == 1 and j == 1:
            seeds.append(1)
        elif i == -1 and j == -1:
            seeds.append(-1)
        else:
            seeds.append(1)
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 and x2', x1, 'and', x2, '=', seeds)
    return seeds
def ternary_or(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    for i, j in zip(x1, x2):
        if i == 0 or j == 0:
            seeds.append(0)
        elif i == 1 and j == 1:
            seeds.append(1)
        elif i == -1 and j == -1:
            seeds.append(-1)
        else:
            seeds.append(-1)
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 or x2', x1, 'or', x2, '=', seeds)
    return seeds
def ternary_reverse(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    if x1 == x2:
        for i in x1:
            seeds.append(i)
    else:
        for i, j in zip(x1, x2):
            if i == -1 and j == 0:
                seeds.append(1)
            elif i == -1 and j == 1:
                seeds.append(0)
            elif i == 0 and j == 1:
                seeds.append(-1)
            elif i == 0 and j == -1:
                seeds.append(1)
            elif i == 1 and j == -1:
                seeds.append(0)
            elif i == 1 and j == 0:
                seeds.append(-1)
            elif i == 1 and j == 1:
                seeds.append(random.choice([-1, 0]))
            elif i == 0 and j == 0:
                seeds.append(random.choice([-1, 1]))
            elif i == -1 and j == -1:
                seeds.append(random.choice([0, 1]))
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 or x2', x1, 'or', x2, '=', seeds)
    return seeds
def ternary_oddeven(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    for i in range(len(x1)):
        if (i % 2) == 0:
            seeds.append(x1[i])
        else:
            seeds.append(x2[i])
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 or x2', x1, 'or', x2, '=', seeds)
    return seeds
def ternary_halfhalf(x1, x2):
    seeds = []
    x1.reverse()
    x2.reverse()
    for i in range(len(x1)):
        if i <= len(x1)/2:
            seeds.append(x1[i])
        else:
            seeds.append(x2[i])
    seeds.reverse()
    x1.reverse()
    x2.reverse()
    # print('x1 or x2', x1, 'or', x2, '=', seeds)
    return seeds


#  根据生成的表达式计算树中每个结点（包括叶子结点）的值
def generate_tree(individual, total_list, input_names_list):
    individual = individual.replace('(', ' ').replace(', ', ' ').replace(')', '')
    individual_list = individual.split(' ')
    node_bank = []
    for j in range(len(individual_list)):
        for inputs_names in input_names_list:
            if individual_list[j] == inputs_names:
                individual_list[j] = total_list[input_names_list.index(inputs_names)][0]
                node_bank.append(total_list[input_names_list.index(inputs_names)][0])  # 添加原有的变量（即不是经过运算符计算得来的结点）
    restart = True
    while restart:
        restart = False
        for i in range(len(individual_list)):
            if isinstance(individual_list[i], str):
                if not isinstance(individual_list[i + 1], str):
                    if not isinstance(individual_list[i + 2], str):
                        if individual_list[i] == "ternary_add":
                            new_node = ternary_add(individual_list[i + 1], individual_list[i + 2])
                        elif individual_list[i] == "ternary_sub":
                            new_node = ternary_sub(individual_list[i + 1], individual_list[i + 2])
                        elif individual_list[i] == "ternary_mul":
                            new_node = ternary_mul(individual_list[i + 1], individual_list[i + 2])
                        elif individual_list[i] == "ternary_and":
                            new_node = ternary_and(individual_list[i + 1], individual_list[i + 2])
                        elif individual_list[i] == "ternary_or":
                            new_node = ternary_or(individual_list[i + 1], individual_list[i + 2])
                        # elif individual_list[i] == "ternary_reverse":
                        #     new_node = ternary_reverse(individual_list[i + 1], individual_list[i + 2])
                        elif individual_list[i] == "ternary_oddeven":
                            new_node = ternary_oddeven(individual_list[i + 1], individual_list[i + 2])
                        else:
                            new_node = ternary_halfhalf(individual_list[i + 1], individual_list[i + 2])
                        node_bank.append(new_node)
                        individual_list.pop(i)  # 删除运算符
                        individual_list.pop(i)  # 删除x1
                        individual_list.pop(i)  # 删除x2
                        individual_list.insert(i, new_node)
                        restart = True  # 重新遍历
                        break
    # 最后返回的node_bank是一个列表，存储的是一棵树（即一个个体）中的所有结点的值
    return node_bank


# 将矩阵转置
def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix


# 去重
def de_duplication_node_bank(node_bank):
    node_bank_de_duplication = []
    # 去重
    for i in node_bank:
        if not i in node_bank_de_duplication:
            node_bank_de_duplication.append(i)
    # node_bank2 = transpose(node_bank2)  # 转置，并将list类型的矩阵转化为ndarray类型的矩阵
    # matrix = np.array(node_bank2)
    # return matrix
    return node_bank_de_duplication


# 合法性检验
def legality_check(node_bank, n_class):
    flag = True
    # 列的要求
    # 限定编码矩阵的最小列数为4，限定编码矩阵的最大列数2 * n_class
    # if len(node_bank) < 4 or len(node_bank) > 2 * n_class:
    node_bank_temp = []
    if len(node_bank) < n_class:
        flag = False
    for node_new in node_bank:
        node = node_new[2:]
        if not (-1 in node and 1 in node):  # 每一列必须包含-1和1
            flag = False
        # if node == node[::-1]:  # 不能含有相反的列（这里理解错了，应该修改之，不能含有逆向的列）
        #     flag = False
        neg_node = [-x for x in node]
        if node not in node_bank_temp and neg_node not in node_bank_temp:
            node_bank_temp.append(node)
    if len(node_bank) != len(node_bank_temp):
        flag = False

    # 行的要求
    node_bank_transpose_new = transpose(node_bank)
    node_bank_transpose = node_bank_transpose_new[2:]
    node_bank_temp = []
    for node_transpose in node_bank_transpose:
        # 不能含有全是0的行
        if -1 not in node_transpose and 1 not in node_transpose:
            flag = False
        # 不能含有相同的行
        if node_transpose not in node_bank_temp:
            node_bank_temp.append(node_transpose)
    if len(node_bank_transpose) != len(node_bank_temp):
        flag = False
    # matrix = np.array(node_bank2)
    return flag


# 生成个体
def initRepeat(container, func, n, total_list, input_names_list, n_class):
    func_list = []
    i = 1
    while i != n:
        fx = func()
        # print("基因", i, fx)
        node_bank = generate_tree(str(fx), total_list, input_names_list)
        node_bank_de_duplication = de_duplication_node_bank(node_bank)  # 去重
        if legality_check(node_bank_de_duplication, n_class):
            # print("基因", i, fx)
            # print("合法个体", np.array(transpose(node_bank_de_duplication)))
            func_list.append(fx)
            i = i+1
        else:
            i = i
    return func_list


def get_centroid(class_name, train_data, train_label):
    temp_data = copy.deepcopy(train_data)
    temp_label = copy.deepcopy(train_label)
    target_data = [temp_data[i] for i in range(len(temp_label)) if temp_label[i] == class_name]
    return np.average(target_data, axis=0).tolist()


def euclidence_distance(x, y, soft=True):
    if soft:
        x_ = np.array([x[i] for i in range(len(x)) if x[i]!=0 and y[i]!=0])
        y_ = np.array([y[i] for i in range(len(y)) if x[i]!=0 and y[i]!=0])
    else:
        x_ = np.array(x)
        y_ = np.array(y)
    if len(x_)==0:
        raise Exception('len(x_)==0')
    return np.sqrt(np.sum(np.power(x_-y_,2))/(len(x_)+0.0001))


def euclidence_distance2(x, y, soft=True):
    # print("x", x)
    # print("y", y)
    if soft:
        x_ = np.array([x[i] for i in range(len(x)) if x[i]!=0 and y[i]!=0])
        y_ = np.array([y[i] for i in range(len(y)) if x[i]!=0 and y[i]!=0])
    else:
        x_ = np.array(x)
        y_ = np.array(y)
    if len(x_)==0:
        raise Exception('len(x_)==0')
    return np.sqrt(np.sum(np.power(x_-y_,2))/(len(x_)+0.0001))


def purity(count_positive, count_negative):
    return max([count_positive, count_negative]) / (count_positive + count_negative + 0.00001)


def get_tough_classes(conf_matrix, index):
    conf_matrix = np.array(conf_matrix)
    row_sum = np.sum(conf_matrix, axis=1).reshape([-1, 1])
    conf_matrix_ratio = conf_matrix / row_sum
    for i in range(len(conf_matrix_ratio)):
        conf_matrix_ratio[i][i] = -1
    max_index = np.argmax(conf_matrix_ratio)
    row_index = int(max_index / len(conf_matrix_ratio))
    col_index = max_index % len(conf_matrix_ratio)
    return [index[row_index], index[col_index]]


def stratified_sample(data, label):
    temp_data, temp_label = RandomOverSampler().fit_resample(data, label)
    sample_data, _, sample_label, _ = train_test_split(temp_data, temp_label, test_size=0.2, stratify=temp_label)
    return sample_data.tolist(), sample_label.tolist()


def get_misclassified_samples(train_data, train_label, validate_data, validate_label, predicted_label):
    misclassified_data = [validate_data[i] for i in range(len(validate_label)) if
                          validate_label[i] != predicted_label[i]]
    misclassified_label = [validate_label[i] for i in range(len(validate_label)) if
                           validate_label[i] != predicted_label[i]]
    misclassified_label_count = collections.Counter(misclassified_label)
    # print("错分类别统计", misclassified_label_count)
    validate_label_count = collections.Counter(validate_label)
    # print("验证集类别统计", validate_label_count)
    misclassified_ratio = {key: misclassified_label_count[key] / validate_label_count[key] for key in
                           misclassified_label}
    # print("错分类别占比", misclassified_ratio)
    min_ratio_threshold = 0.05
    misclassified_data = [misclassified_data[i] for i in range(len(misclassified_label)) if
                          misclassified_ratio[misclassified_label[i]] >= min_ratio_threshold]
    misclassified_label = [misclassified_label[i] for i in range(len(misclassified_label)) if
                           misclassified_ratio[misclassified_label[i]] >= min_ratio_threshold]
    # print(len(train_data))
    # train_data, train_label = stratified_sample(train_data, train_label)
    # print(len(train_data))
    train_label_count = collections.Counter(train_label)
    res_data = []
    res_label = []
    for l in misclassified_label_count:
        neighbor_number = int(np.ceil(misclassified_ratio[l] * train_label_count[l]))
        neighbor_per_sample = int(np.ceil(neighbor_number / misclassified_label_count[l]))
        temp_train_data = [train_data[i] for i in range(len(train_label)) if train_label[i] == l]
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(temp_train_data)
        for i in range(len(misclassified_label)):
            if misclassified_label[i] == l:
                neighbors = nn.kneighbors([misclassified_data[i]], n_neighbors=neighbor_per_sample,
                                          return_distance=False)
                res_data += [temp_train_data[i] for i in neighbors[0]]
                res_label += [l] * len(neighbors[0])
    res_data = np.array(res_data)
    res_label = np.array(res_label)
    # print("train_x", len(res_data))
    # print("train_y", len(res_label))
    # print("train_y", np.unique(res_label))
    return res_data, res_label


def get_class_data(data, label, class_name):
    if class_name not in label:
        return []
    result_data = [data[i] for i in range(len(label)) if label[i] == class_name]
    result_label = [label[i] for i in range(len(label)) if label[i] == class_name]
    return result_data, result_label


def make_column_with_centroid_class(centroid_classes=(), centroids=None, custom_data=None, custom_label=None, index=None):
    column = [0] * len(index)
    if len(centroid_classes) > 0:
        column[index.index(centroid_classes[0])] = 1
        positive_centroid = get_centroid(centroid_classes[0], custom_data, custom_label)
        column[index.index(centroid_classes[1])] = -1
        negative_centroid = get_centroid(centroid_classes[1], custom_data, custom_label)
    else:
        positive_centroid = centroids[0]
        negative_centroid = centroids[1]
    temp_data = np.array(custom_data).tolist()
    temp_label = copy.deepcopy(custom_label)
    column2 = copy.deepcopy(column)
    for i, c in enumerate(index):
        if c not in centroid_classes:
            temp_class_data = copy.deepcopy([temp_data[i] for i in range(len(temp_label)) if temp_label[i] == c])
            positive_count = 0
            negative_count = 0
            for data in temp_class_data:
                if euclidence_distance(data, positive_centroid, soft=False) < euclidence_distance(data, negative_centroid, soft=False):
                    positive_count += 1
                else:
                    negative_count += 1
            # print("纯度", purity(positive_count, negative_count))
            if purity(positive_count, negative_count) >= 0.6:
                column[i] = 1 if positive_count > negative_count else -1
    return column


def train_base_classifier(column, estimator_type=None, custom_data=None, custom_label=None, fill_zero=True, indexs=None):
    train_data = copy.deepcopy(custom_data)
    train_data = np.array(train_data).tolist()
    train_label = copy.deepcopy(custom_label)
    # train_data, train_label = stratified_sample(train_data, train_label)
    negative_class = []
    positive_class = []
    for index, value in enumerate(column):
        if value == 1:
            positive_class.append(indexs[index])
        elif value == -1:
            negative_class.append(indexs[index])
    data = []
    label = []
    for index, value in enumerate(train_label):
        if value in positive_class:
            data.append(train_data[index])
            label.append(1)
        elif value in negative_class:
            data.append(train_data[index])
            label.append(-1)
    classifier = get_base_clf(estimator_type).fit(data, label)
    return column, classifier


def validate_single_classifier(column, classifier, custom_data=None, custom_label=None, indexs=None):
    temp_data = np.array(custom_data).tolist()
    temp_label = list(custom_label)
    temp_data = [temp_data[i] for i in range(len(temp_label)) if column[indexs.index(temp_label[i])] != 0]
    temp_label = [temp_label[i] for i in range(len(temp_label)) if column[indexs.index(temp_label[i])] != 0]
    temp = []
    for l in temp_label:
        temp.append(column[indexs.index(l)])
    temp_label = temp
    predicted_label = classifier.predict(temp_data)
    result = Evaluation(temp_label, predicted_label).evaluation(accuracy=True, precision=True, sensitivity=True, Fscore=True)
    accuracy = result['accuracy']
    precision = result['precision']
    sensitivity = result['sensitivity']
    Fscore = result['Fscore']
    return accuracy


def predict(data, classifiers, index, classifier_recall, node_bank_de_duplication_T):
    row_code = []
    # print("共有", len(classifiers), "个分类器")
    for i, classifier in enumerate(classifiers):
        temp_data = data
        temp_code = classifier.predict(temp_data).tolist()
        # print("temp_code", temp_code)
        row_code.append(temp_code)
    matrix = transpose(node_bank_de_duplication_T)
    # print("预测score矩阵共有", len(row_code), "个列")
    # print("code_matrix", matrix)
    # print("编码矩阵共有", len(node_bank_de_duplication_T), "个列")
    row_code = np.array(row_code).T.tolist()
    predicted_label = []
    for code in row_code:
        temp_label = None
        min_distance = np.inf
        for i, matrix_code in enumerate(matrix):
            distance = euclidence_distance2(code, matrix_code)
            if distance < min_distance:
                min_distance = distance
                temp_label = index[i]
        predicted_label.append(temp_label)
    return predicted_label


# 特征选择
def process_select_svmrfe30(data, labels):
    feature_num = data.shape[1]
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=feature_num, step=2)
    rfe.fit(data, labels)

    # selected = np.arange(features.shape[1])[rfe.support_]
    selected = np.argsort(rfe.ranking_)
    feature_number = int(feature_num * 0.3)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X
def process_select_forest30(data, labels):
    feature_num = data.shape[1]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(data, labels)

    selected = np.argsort(forest.feature_importances_)[::-1]
    feature_number = int(feature_num * 0.3)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X
def process_select_bsswss30(data, labels):
    def bss_wss_value(f, labels):
        names = sorted(set(labels))
        wss, bss = np.array([]), np.array([])
        for name in names:
            f_k = f[labels == name]
            f_m = f_k.mean()
            d_m = (f_m - f.mean()) ** 2
            d_z = (f_k - f_m) ** 2
            bss = np.append(bss, d_m.sum())
            wss = np.append(wss, d_z.sum())
        z, m = bss.sum(), wss.sum()
        bsswss = z / m if m > 0 else 0
        return bsswss

    feature_num = data.shape[1]
    i = 0
    x, y = [], []
    for f in data.transpose():
        x.append(i)
        y.append(bss_wss_value(f, labels))
    selected = np.argsort(y)[::-1]
    feature_number = int(feature_num * 0.3)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X

def process_select_svmrfe50(data, labels):
    feature_num = data.shape[1]
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=feature_num, step=2)
    rfe.fit(data, labels)

    # selected = np.arange(features.shape[1])[rfe.support_]
    selected = np.argsort(rfe.ranking_)
    feature_number = int(feature_num * 0.50)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X
def process_select_forest50(data, labels):
    feature_num = data.shape[1]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(data, labels)

    selected = np.argsort(forest.feature_importances_)[::-1]
    feature_number = int(feature_num * 0.50)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X
def process_select_bsswss50(data, labels):
    def bss_wss_value(f, labels):
        names = sorted(set(labels))
        wss, bss = np.array([]), np.array([])
        for name in names:
            f_k = f[labels == name]
            f_m = f_k.mean()
            d_m = (f_m - f.mean()) ** 2
            d_z = (f_k - f_m) ** 2
            bss = np.append(bss, d_m.sum())
            wss = np.append(wss, d_z.sum())
        z, m = bss.sum(), wss.sum()
        bsswss = z / m if m > 0 else 0
        return bsswss

    feature_num = data.shape[1]
    i = 0
    x, y = [], []
    for f in data.transpose():
        x.append(i)
        y.append(bss_wss_value(f, labels))
    selected = np.argsort(y)[::-1]
    feature_number = int(feature_num * 0.50)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X

def process_select_svmrfe70(data, labels):
    feature_num = data.shape[1]
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=feature_num, step=2)
    rfe.fit(data, labels)

    # selected = np.arange(features.shape[1])[rfe.support_]
    selected = np.argsort(rfe.ranking_)
    feature_number = int(feature_num * 0.7)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X
def process_select_forest70(data, labels):
    feature_num = data.shape[1]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(data, labels)

    selected = np.argsort(forest.feature_importances_)[::-1]
    feature_number = int(feature_num * 0.7)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X
def process_select_bsswss70(data, labels):
    def bss_wss_value(f, labels):
        names = sorted(set(labels))
        wss, bss = np.array([]), np.array([])
        for name in names:
            f_k = f[labels == name]
            f_m = f_k.mean()
            d_m = (f_m - f.mean()) ** 2
            d_z = (f_k - f_m) ** 2
            bss = np.append(bss, d_m.sum())
            wss = np.append(wss, d_z.sum())
        z, m = bss.sum(), wss.sum()
        bsswss = z / m if m > 0 else 0
        return bsswss

    feature_num = data.shape[1]
    i = 0
    x, y = [], []
    for f in data.transpose():
        x.append(i)
        y.append(bss_wss_value(f, labels))
    selected = np.argsort(y)[::-1]
    feature_number = int(feature_num * 0.7)
    selected = selected[:feature_number]
    selected_list = selected.tolist()
    X = data[:, selected_list]
    return X


# 局部优化
def local_optimization(validate_y, pred_label_M, node_bank_de_duplication, Fscore, accuracy, n_class, estimator_type, train_x_svmrfe30, train_y_svmrfe30, train_x_rf30, train_y_rf30, train_x_bsswss30, train_y_bsswss30, validate_x_svmrfe30, validate_x_rf30, validate_x_bsswss30, train_x_svmrfe50, train_y_svmrfe50, train_x_rf50, train_y_rf50, train_x_bsswss50, train_y_bsswss50, validate_x_svmrfe50, validate_x_rf50, validate_x_bsswss50, train_x_svmrfe70, train_y_svmrfe70, train_x_rf70, train_y_rf70, train_x_bsswss70, train_y_bsswss70, validate_x_svmrfe70, validate_x_rf70, validate_x_bsswss70, index, file_name, file_path):
    node_bank_de_duplication_T = copy.deepcopy(node_bank_de_duplication)
    add_num = 0
    Fscore_new = Fscore
    Acc_new = accuracy
    old_Per_Cla_Acc = []
    tough_classes_list = []
    f2 = open("./Logging/" + file_path + "/" + file_name + "迭代结果.txt", 'a+')
    print("局部优化前的原始编码矩阵的Fscore值：", Fscore_new, file=f2)
    print("局部优化前的原始编码矩阵的Accuracy值：", Acc_new, file=f2)
    flag = True
    pred_label_T = pred_label_M
    CMs = confusion_matrix(validate_y, pred_label_T)
    print("局部优化前的原始编码矩阵对应的混淆矩阵", file=f2)
    print(CMs, file=f2)
    # print("实际类别", validate_y)
    # print("预测类别", pred_label_T)
    # 创造新列
    # print("总的类别", index)
    # print("*****************")
    CMlist = CMs.tolist()
    for i in range(len(CMlist)):
        total = sum(CMlist[i])
        if total != 0:
            acc = CMlist[i][i] / total
        else:
            acc = 0
        print("局部优化前的原始矩阵的第", index[i], "类别的正确率为", acc, file=f2)
        old_Per_Cla_Acc.append(acc)
    print("局部优化前的原始编码矩阵的分类准确率为：", old_Per_Cla_Acc, file=f2)
    cm_matrix = []
    for i in range(len(CMlist)):
        cm_row = []
        for j in range(len(CMlist[i])):
            total = sum(CMlist[i])
            if total != 0:
                cm_ij = CMlist[i][i] / total
            else:
                cm_ij = 0
            cm_row.append(cm_ij)
        cm_matrix.append(cm_row)
    # print(np.array(cm_matrix))  #混淆准确率矩阵
    lm_matrix = []
    for i in range(len(cm_matrix)):
        lm_row = []
        for j in range(len(cm_matrix[i])):
            lm_ij = 0
            for k in range(len(cm_matrix)):
                lm_ij += (cm_matrix[i][k] - cm_matrix[j][k]) ** 2
            lm_row.append(lm_ij)
        lm_matrix.append(lm_row)
    for i in range(len(lm_matrix)):
        for j in range(len(lm_matrix[i])):
            if i > j:
                lm_matrix[i][j] = 0
    print("原始混淆矩阵对应的相似性度量矩阵", file=f2)
    print(np.array(lm_matrix), file=f2)  # 相似性度量矩阵
    lm_all = []
    for r in range(len(lm_matrix)):
        for c in range(len(lm_matrix[r])):
            if r < c:
                lm_one = []
                lm_one.extend([r, c, lm_matrix[r][c] + lm_matrix[c][r]])
                lm_all.append(lm_one)
    print("", file=f2)
    print("开始新增列", file=f2)
    new_Per_Cla_Acc = []
    while flag:
        print("~~~~~~~~~~~~~~~~~~", file=f2)
        new_Per_Cla_Acc.clear()
        print("第", add_num+1, "次新增列", file=f2)
        fs_list = [-1, 0, 1]
        fs_one = random.choice(fs_list)
        fs_num_list = [-1, 0, 1]
        fs_num_one = random.choice(fs_num_list)
        if fs_one == -1:
            if fs_num_one == -1:
                train_data, train_label = train_x_svmrfe30, train_y_svmrfe30
                validate_x = validate_x_svmrfe30
            elif fs_num_one == 0:
                train_data, train_label = train_x_svmrfe50, train_y_svmrfe50
                validate_x = validate_x_svmrfe50
            else:
                train_data, train_label = train_x_svmrfe70, train_y_svmrfe70
                validate_x = validate_x_svmrfe70
        elif fs_one == 0:
            if fs_num_one == -1:
                train_data, train_label = train_x_rf30, train_y_rf30
                validate_x = validate_x_rf30
            elif fs_num_one == 0:
                train_data, train_label = train_x_rf50, train_y_rf50
                validate_x = validate_x_rf50
            else:
                train_data, train_label = train_x_rf70, train_y_rf70
                validate_x = validate_x_rf70
        else:
            if fs_num_one == -1:
                train_data, train_label = train_x_bsswss30, train_y_bsswss30
                validate_x = validate_x_bsswss30
            elif fs_num_one == 0:
                train_data, train_label = train_x_bsswss50, train_y_bsswss50
                validate_x = validate_x_bsswss50
            else:
                train_data, train_label = train_x_bsswss70, train_y_bsswss70
                validate_x = validate_x_bsswss70
        train_x, train_y = get_misclassified_samples(train_data, train_label, validate_x, validate_y, pred_label_T)  # 获得难分样本，形成新的训练数据集
        # CMs = confusion_matrix(validate_y, pred_label_T)
        lm_all_new = []
        for i in range(len(lm_all)):
            lm_all_new.append(lm_all[i][2])
        min_index = lm_all_new.index(min(lm_all_new))
        cla_i = lm_all[min_index][0]
        cla_j = lm_all[min_index][1]
        tough_classes = [index[cla_i], index[cla_j]]
        print("难分类别对为：", tough_classes, file=f2)
        # tough_classes2 = get_tough_classes(CMs, index)  # 获得难分类别
        for tough_class in tough_classes:  # 如果难分样本中没有难分类别类，则从原始训练数据中随机选取一些添加进去
            if tough_class not in np.unique(train_y):
                temp_train_data, temp_train_label = get_class_data(train_data, train_label, tough_class)
                train_x = train_x.tolist()
                train_y = train_y.tolist()
                train_x += random.sample(temp_train_data, k=int(np.ceil(0.5 * len(temp_train_data))))
                train_y += random.sample(temp_train_label, k=int(np.ceil(0.5 * len(temp_train_label))))
                train_x = np.array(train_x)
                train_y = np.array(train_y)
        # 根据难分类别即难分样本生成额外的列
        column = make_column_with_centroid_class(centroid_classes=tough_classes, custom_data=train_x, custom_label=train_y, index=index)

        # 使用难分样本对当前列进行训练
        h, classifier = train_base_classifier(column, estimator_type=estimator_type, custom_data=train_x, custom_label=train_y, fill_zero=True, indexs=index)
        h_acc = validate_single_classifier(column, classifier, custom_data=validate_x, custom_label=validate_y, indexs=index)
        # print("新生成列为：", column)
        # for index, value in enumerate(column):
        #     if value == 0:
        #         column[index] = -100
        h_new = column
        # print("GEPv7.1的难分类别", tough_classes[::-1])
        # print("新加列", h_new)
        # print("特征位", fs_one)
        h = [fs_one] + [fs_num_one] + h_new
        # print("新加列的准确率为：", h_acc)
        if h_acc >= 0.6:
            node_bank_de_duplication_T.append(h)
            # 若加入新列使编码矩阵不合法，则重新生成
            if legality_check(node_bank_de_duplication_T, n_class):
                node_bank_final_T = transpose(node_bank_de_duplication_T)  # 转置，并将list类型的矩阵转化为ndarray类型的矩阵
                matrix_T = np.array(node_bank_final_T)
                Code_Matrix_T = matrix_T  # M是函数返回值即matrix（编码矩阵）
                estimator = get_base_clf(estimator_type)
                sec_T = SimpleECOCClassifier2(estimator, Code_Matrix_T)
                sec_T.fit(train_x_svmrfe30, train_y_svmrfe30, train_x_rf30, train_y_rf30, train_x_bsswss30, train_y_bsswss30, train_x_svmrfe50, train_y_svmrfe50, train_x_rf50, train_y_rf50, train_x_bsswss50, train_y_bsswss50, train_x_svmrfe70, train_y_svmrfe70, train_x_rf70, train_y_rf70, train_x_bsswss70, train_y_bsswss70)
                pred_label_T = sec_T.predict(validate_x_svmrfe30, validate_x_rf30, validate_x_bsswss30, validate_x_svmrfe50, validate_x_rf50, validate_x_bsswss50, validate_x_svmrfe70, validate_x_rf70, validate_x_bsswss70)
                T_fscore = f1_score(validate_y, pred_label_T, average="macro")
                T_acc = accuracy_score(validate_y, pred_label_T)
                # print("当前列数：", len(node_bank_de_duplication))
                # print("优化后矩阵性能", T_fscore)
                if T_fscore > Fscore:
                    # print("新增列成功")
                    tough_classes_list.append(tough_classes)
                    # print("当前矩阵的难分类对为：", tough_classes_list, file=f2)
                    print("新加列为：", h, file=f2)
                    Fscore_new = T_fscore
                    Acc_new = T_acc
                    print("加入新增列后当前矩阵为：", file=f2)
                    print(matrix_T, file=f2)
                    print("加入新增列后当前矩阵的Fscore值为：", T_fscore, file=f2)
                    print("加入新增列后当前矩阵的accuracy值为：", T_acc, file=f2)
                    CMs_T = confusion_matrix(validate_y, pred_label_T)
                    CMlist_T = CMs_T.tolist()
                    for i in range(len(CMlist_T)):
                        total = sum(CMlist_T[i])
                        if total != 0:  # 可能会存在没有这一类的情况，这会使得混淆矩阵某行全为0
                            acc = CMlist_T[i][i] / total
                        else:
                            acc = 0
                        print("当前矩阵的第", index[i], "类别的正确率为", acc, file=f2)
                        new_Per_Cla_Acc.append(acc)
                    print("", file=f2)
                    # avg_col_value.append(h_fscore)
                else:
                    node_bank_de_duplication_T.pop()
                    print("不加入该列", file=f2)
                    print("", file=f2)
                # 返回评价值（目标是使其最大化）
                # print("当前列数：", len(node_bank_de_duplication))
                # print()
            else:
                node_bank_de_duplication_T.pop()
                print("不加入该列", file=f2)
                print("", file=f2)
        add_num += 1
        limit = 2 * n_class - len(node_bank_de_duplication_T)
        # limit = 5 * math.ceil(np.log2(n_class) - len(node_bank_de_duplication))
        if add_num > limit:
            break
        lm_all[min_index][2] = 1000
    node_bank_de_duplication_T = de_duplication_node_bank(node_bank_de_duplication_T)
    print("~~~~~~~~~~~~~~~~~~", file=f2)
    print("局部优化后的最终矩阵为：", file=f2)
    print(np.array(transpose(node_bank_de_duplication_T)), file=f2)
    print("局部优化后的最终矩阵的Fscore值为：", Fscore_new, file=f2)
    print("局部优化后的最终矩阵的accuracy值为：", Acc_new, file=f2)
    print("局部优化后的最终矩阵的分类准确率为：", new_Per_Cla_Acc, file=f2)  #若为空则说明此次没有进行局部优化",
    print("*****************", file=f2)
    print("", file=f2)
    f2.close()
    return Fscore_new, Acc_new, node_bank_de_duplication_T, old_Per_Cla_Acc, new_Per_Cla_Acc, tough_classes_list
