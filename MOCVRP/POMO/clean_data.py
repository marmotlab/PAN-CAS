import numpy as np
import pandas as pd
import os

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def get_non_dominated(population, final_pop=False):
    """
    Return all non dominated solutions of the population
    :param population: list<{:class:`~moead_framework.solution.one_dimension_solution.OneDimensionSolution`}>
    :return: population: list<{:class:`~moead_framework.solution.one_dimension_solution.OneDimensionSolution`}>
    """
#     arr = []
#     for s in population:
#         arr.append(s.F)

    # new_pop = list((population[i],i) for i in is_pareto_efficient(population, return_mask=False))
    # new_pop = list((population[i]) for i in is_pareto_efficient(population, return_mask=True))
    if final_pop:
        new_pop = list((population[i]) for i in is_pareto_efficient(population, return_mask=False))
        return new_pop
    else:
        pop_idx = is_pareto_efficient(population, return_mask=True)
        return pop_idx

def weakly_dominates(left, right):
    return all((x <= y for x, y in zip(left, right)))

def dominates(left, right):
    if (left == right).all(): return False
    return weakly_dominates(left, right)


def remove_dominated(nds):
    filtered = []
    for nd in nds:
        if not any( (dominates(other, nd) for other in nds) ):
            filtered.append(nd)
    return filtered

if __name__ == "__main__":
    filename = "objective_4KP-100-4.lp_2000.csv"
    dir_name = 'experiment/time2000/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    csv_data1 = pd.read_csv(filename, low_memory=False)
    csv_df1 = np.array(pd.DataFrame(csv_data1)).astype(int)
    population = -csv_df1[:, 1:]
    # time = np.zeros((1, 6)).astype(int)
    # time[0, 0] = csv_df1[-1, 1]
    # print(time)
    new_pop = np.array(get_non_dominated(population))
    # new_solution = np.concatenate((new_pop, time), axis=0)
    # print(new_solution[-1])
    np.save('{}objective_4KP_100_4.npy'.format(dir_name), new_pop)
    print(new_pop)
