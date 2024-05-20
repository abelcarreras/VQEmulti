import numpy as np

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

def delete_an_operator(einstein_index, coefficients):
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
