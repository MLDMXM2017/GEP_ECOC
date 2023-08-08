# coding=utf-8
import deap
from deap import creator, base, tools
import random
import warnings
import customize_tools as ct
import copy


def _validate_basic_toolbox(tb):
    """
    Validate the operators in the toolbox *tb* according to our conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a 'select' operator."
    # whether the ops in .pbs are all registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Operators must start with 'mut' or 'cx' except selection."
        assert hasattr(tb, op), "Probability for a operator called '{}' is specified, but this operator is not " \
                                "registered in the toolbox.".format(op)
    # whether all the mut_ and cx_ operators have their probabilities assigned in .pbs
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('{0} is registered, but its probability is NOT assigned in Toolbox.pbs. '
                          'By default, the probability is ZERO and the operator {0} will NOT be applied.'.format(op),
                          category=UserWarning)


def _apply_modification(population, operator, pb, total_list=None, input_names_list=None, n_class=0):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    population_copy = copy.deepcopy(population)
    population_copy_new = []
    for i in range(len(population_copy)):
        mutation_num = 1
        flag = True
        if random.random() < pb:
            while flag:
                population_copy_temp = []
                population_copy_temp.append(population_copy[i])
                population_copy_temp2 = copy.deepcopy(population_copy_temp)
                # print("开始变异， 变异次数：", mutation_num)
                if mutation_num <= 100:
                    # print("基因", i, "变异前", str(population_copy_temp[0]))
                    population_copy_temp[0], = operator(population_copy_temp2[0])
                    # print("基因", i, "变异后", str(population_copy_temp[0]))
                    node_bank = ct.generate_tree(str(population_copy_temp[0]), total_list=total_list, input_names_list=input_names_list)
                    node_bank_de_duplication = ct.de_duplication_node_bank(node_bank)  # 去重
                    if ct.legality_check(node_bank_de_duplication, n_class):
                        # print("基因", i, "变异合法")
                        population_copy_new.append(population_copy_temp[0])
                        del population_copy_new[0].fitness.values
                        flag = False
                    else:
                        mutation_num += 1
                        # print("population", population[i])
                        # print("population_copy", population_copy[i])
                        # del population_copy_temp[0]
                        population_copy_temp.pop(0)
                        # print("基因", i, "变异非法")
                else:
                    # population_copy[i] = population[i]
                    population_copy_new.append(population[i])
                    flag = False
        else:
            population_copy_new.append(population[i])
    return population_copy_new


def _apply_crossover(population, operator, pb, total_list=None, input_names_list=None, n_class=0):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    population_copy = copy.deepcopy(population)
    population_copy_new = []
    for i in range(1, len(population), 2):
        crossover_num = 1
        flag = True
        if random.random() < pb:
            while flag:
                population_copy_temp = []
                population_copy_temp.append(population_copy[i-1])
                population_copy_temp.append(population_copy[i])
                population_copy_temp2 = copy.deepcopy(population_copy_temp)
                # print("开始交叉， 交叉次数：", crossover_num)
                if crossover_num <= 100:
                    population_copy_temp[0], population_copy_temp[1] = operator(population_copy_temp2[0], population_copy_temp2[1])
                    node_bank1 = ct.generate_tree(str(population_copy_temp[0]), total_list=total_list, input_names_list=input_names_list)
                    node_bank2 = ct.generate_tree(str(population_copy_temp[1]), total_list=total_list, input_names_list=input_names_list)
                    node_bank_de_duplication1 = ct.de_duplication_node_bank(node_bank1)  # 去重
                    node_bank_de_duplication2 = ct.de_duplication_node_bank(node_bank2)  # 去重

                    if ct.legality_check(node_bank_de_duplication1 ,n_class) and ct.legality_check(node_bank_de_duplication2 ,n_class):
                        # print("基因", i, "交叉合法")
                        population_copy_new.append(population_copy_temp[0])
                        population_copy_new.append(population_copy_temp[1])
                        del population_copy_temp[0].fitness.values
                        del population_copy_temp[1].fitness.values
                        flag = False
                    else:
                        crossover_num += 1
                        population_copy_temp.clear()
                        # population_copy_temp.pop(0)
                        # population_copy_temp.pop(1)
                        # print("基因", i, "交叉非法")
                else:
                    population_copy_new.append(population[i-1])
                    population_copy_new.append(population[i])
                    flag = False
        else:
            population_copy_new.append(population[i-1])
            population_copy_new.append(population[i])
    return population_copy_new


def gep_simple(population, toolbox, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__, total_list=None, input_names_list=None, n_class=0, file_name=None, file_path=None):
    n_class = n_class
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # 进化迭代
    f1 = open("./Logging/" + file_path + "/" + file_name + "迭代结果.txt", 'w+')
    print("遗传算法的迭代开始", file=f1)
    f1.close()
    for gen in range(n_generations):
        f2 = open("./Logging/" + file_path + "/" + file_name + "迭代结果.txt", 'a+')
        print("", file=f2)
        print("*****************", file=f2)
        print("第", gen+1, "代：", file=f2)
        print("*****************", file=f2)
        f2.close()
        # 只评估无效的，即不需要重新评估未更改的
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        # individuals = [ind for ind in population]
        # fitnesses = toolbox.map(toolbox.evaluate, individuals)
        # for pop in population:
            # print("ssss", fit)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)
        if verbose:
            print(logbook.stream)
        if gen == n_generations:
            break

        # selection with elitism
        elites = deap.tools.selBest(population, k=n_elites)
        offspring = toolbox.select(population, len(population) - n_elites)
        # replication
        offspring = [toolbox.clone(ind) for ind in offspring]
        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op], total_list=total_list, input_names_list=input_names_list, n_class=n_class)
        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op], total_list=total_list, input_names_list=input_names_list, n_class=n_class)

        # replace the current population with the offsprings
        population = elites + offspring
    return population, logbook


__all__ = ['gep_simple']


