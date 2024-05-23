import numpy as np
from copy import deepcopy
def generate_new_operator(operators_pool, einstein_index, cheat_param):
    # I need to asign a 'fitness' to each element of the pool
    fitness = []
    indices_operator_pool = []
    for i in range(len(operators_pool)):
        indices_operator_pool.append(i)
    for i in range(len(operators_pool)):
        count = 1
        for j in range(len(einstein_index)):
            if einstein_index[j] == indices_operator_pool[i]:
                count += 1
        fitness.append(count)
    def makeWheel(cheat_param):
        wheel = []
        init_point = 0
        points = []

        for i in range(len(fitness)):
            points.append(cheat_param*(1/fitness[i])+(1/len(fitness)))
        points_sum = np.sum(points)
        for i in range(len(fitness)):
            points[i] = points[i]/points_sum

        for i in range(len(fitness)):
            f = points[i]
            wheel.append((init_point, init_point + f, i))
            init_point += f
        return wheel

    # Here we generate the random position of the first pointer
    r = np.random.rand()

    # Create the wheel and store the chosen peasants in parent_population
    wheel = makeWheel(cheat_param)
    for j in range(len(wheel)):
        if wheel[j][0] <= r < wheel[j][1]:
            selected = wheel[j][2]



    return selected

def delete_an_operator(einstein_index, coefficients, deleted_already):
    delete_probability = []
    indices_deleted = []

    for i in range(len(einstein_index)):
        for j in range(len(deleted_already)):
            if deleted_already[j] == einstein_index[i]:
                indices_deleted.append(i)

    for i in range(len(coefficients)):
        delete_probability.append(abs(1 / coefficients[i]))
        for j in range(len(indices_deleted)):
            if indices_deleted[j] == i:
                delete_probability[i] = 0
    print('deleted_already', deleted_already)
    print(delete_probability)

    #normalize this vector
    total_sum = np.sum(delete_probability)
    for i in range(len(coefficients)):
        delete_probability[i] = delete_probability[i]/total_sum
    print('normalized delete prob', delete_probability)
    def makeWheel(delete_probability):
        wheel = []
        init_point = 0
        for i in range(len(delete_probability)):
            f = delete_probability[i]
            wheel.append((init_point, init_point + f, i))
            init_point += f
        return wheel

    wheel = makeWheel(delete_probability)
    # Here we generate the random position of the first pointer
    r = np.random.rand()
    for j in range(len(wheel)):
        if wheel[j][0] <= r < wheel[j][1]:
            selected = wheel[j][2]

    return selected


def delete_an_operator_change(einstein_index, coefficients):
    delete_probability = []
    for i in range(len(coefficients)):
        delete_probability.append(1/coefficients[i])

    #normalize this vector
    total_sum = np.sum(delete_probability)
    for i in range(len(coefficients)):
        delete_probability[i] = delete_probability[i]/total_sum

    def makeWheel(delete_probability):
        wheel = []
        init_point = 0
        for i in range(len(delete_probability)):
            f = delete_probability[i]
            wheel.append((init_point, init_point + f, i))
            init_point += f
        return wheel

    wheel = makeWheel(delete_probability)
    # Here we generate the random position of the first pointer
    r = np.random.rand()
    for j in range(len(wheel)):
        if wheel[j][0] <= r < wheel[j][1]:
            selected = wheel[j][2]

    return selected



def generate_reduced_pool(operators_pool, selected_already, einstein_index, cheat_param):
    # I need to asign a 'fitness' to each element of the pool
    fitness = []
    indices_operator_pool = []
    for i in range(len(operators_pool)):
        indices_operator_pool.append(i)
    for i in range(len(operators_pool)):
        count = 1
        for j in range(len(einstein_index)):
            if einstein_index[j] == indices_operator_pool[i]:
                count += 1
        fitness.append(count)
    def makeWheel(cheat_param):
        wheel = []
        init_point = 0
        points = []

        for i in range(len(fitness)):
            points.append(cheat_param*(1/fitness[i])+(1/len(fitness)))


        for i in range(len(selected_already)):
            points[selected_already[i]] = 0

        points_sum = np.sum(points)
        for i in range(len(fitness)):
            points[i] = points[i]/points_sum

        for i in range(len(fitness)):
            f = points[i]
            wheel.append((init_point, init_point + f, i))
            init_point += f
        return wheel

    # Here we generate the random position of the first pointer
    r = np.random.rand()

    # Create the wheel and store the chosen peasants in parent_population
    wheel = makeWheel(cheat_param)
    for j in range(len(wheel)):
        if wheel[j][0] <= r < wheel[j][1]:
            selected = wheel[j][2]



    return selected


def interchange_selection(einstein_index, permutations_chosen, coefficients, operators_pool):


    done = 'NO'
    einstein_index_game = deepcopy(einstein_index)
    selected_all = []
    while done == 'NO':
        count = 0
        select_probability = []
        for i in range(len(coefficients)):
            select_probability.append(3.5 - coefficients[i])
            if i <5:
                select_probability[i] = 0
            for j in range(len(selected_all)):
                if selected_all[j] == einstein_index[i]:
                    select_probability[i] = 0
        for i in range(len(select_probability)):
            if select_probability[i] == 0:
                count +=1
        if count == len(select_probability):
            return 0, 2

        #normalize this vector
        total_sum = np.sum(select_probability)
        if total_sum == 0:
            return 0, 2

        for i in range(len(coefficients)):
            select_probability[i] = select_probability[i]/total_sum

        def makeWheel(select_probability):
            wheel = []
            init_point = 0
            for i in range(len(select_probability)):
                f = select_probability[i]
                wheel.append((init_point, init_point + f, i))
                init_point += f
            return wheel

        wheel = makeWheel(select_probability)
        # Here we generate the random position of the first pointer
        r = np.random.rand()
        for j in range(len(wheel)):
            if wheel[j][0] <= r < wheel[j][1]:
                first_selected = wheel[j][2]

        selected_all.append(einstein_index[first_selected])


        select_probability_second = []

        for i in range(len(coefficients)):
            select_probability_second.append(3.5 - coefficients[i])
            if i < 5:
                select_probability_second[i] = 0
            if einstein_index_game[i] == einstein_index_game[first_selected]:
                select_probability_second[i] = 0
            for j in range(len(permutations_chosen)):
                if permutations_chosen[j][0] == einstein_index[first_selected]:
                    if einstein_index_game[i] == permutations_chosen[j][1]:
                        select_probability_second[i] = 0



        # normalize this vector
        total_sum = np.sum(select_probability_second)

        if total_sum == 0:
            continue
        for i in range(len(select_probability_second)):
            select_probability_second[i] = select_probability_second[i] / total_sum


        def makeWheel(select_probability_second):
            wheel = []
            init_point = 0
            for i in range(len(select_probability_second)):
                f = select_probability_second[i]
                wheel.append((init_point, init_point + f, i))
                init_point += f
            return wheel

        wheel = makeWheel(select_probability_second)
        # Here we generate the random position of the first pointer
        r = np.random.rand()
        for j in range(len(wheel)):
            if wheel[j][0] <= r < wheel[j][1]:
                selected_second = wheel[j][2]
        done = 'yes'

    return 1, [first_selected, selected_second]
