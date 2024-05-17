from evo import Evo
import random as rnd
import pandas as pd
import numpy as np

sections = pd.read_csv("sections.csv")
prefs = pd.read_csv("tas.csv")
section_prefs = prefs.loc[:, '0':].values
allo_prefs = prefs["max_assigned"].values


def overallocation(solutions):
    """
    Gets the total amount of over-allocation for sections for all rows
    Param: test: numpy array, one solution
    Return: int, total over-allocated sections in solution for that sol
    """
    overallocated_lst = [sum(lst) - max for lst, max in zip(solutions, allo_prefs) if
                         sum(lst) > max]

    return sum(overallocated_lst)


def no_tas(solutions):
    """
    Calculates number of sections with no TAs assigned
    Param: test: numpy array, one solution
    Return: int, total sections with no tas assigned for that sol
    """
    return sum([1 for section in solutions.T if sum(section) == 0])


def conflicts(solutions):
    """
    Checks for TAs that have been assigned to sections that overlap
    Param: test: numpy array, one solution
    Return: int, total number of conflicts for that sol
    """

    # creates an array where each row is a TA. each of their
    # sections' times are written, and all other sections in that row are 0
    conflict_combs = np.where(solutions == 1, sections['daytime'].to_numpy(), 0)

    # creates a list of conflicting times for each TA
    conflict_list = [[_ for _ in lst if _ != 0] for lst in conflict_combs]

    # creates a set of conflicting times for each TA (i.e. no duplicate values)
    conflict_set = [set([_ for _ in lst if _ != 0]) for lst in conflict_list]

    # adds a 1 to the array for every TA with conflicting assignments
    num_conflicts = [1 for c_lst, c_set in zip(conflict_list, conflict_set) if len(c_lst) != len(c_set)]

    # sums and returns conflict score
    return sum(num_conflicts)


def undersupport(solutions):
    """
    Checks for sections that are below the minimum amount of Tas assigned for it
    Param: solutions: numpy array, one solution
    Return: int, total number of sections that are undersupported for that sol
    """
    undersupport_lst = [min - sum(lst) for lst, min in zip(solutions.T.tolist(), sections['min_ta'].values) if
                        sum(lst) < min]

    return sum(undersupport_lst)


def unwilling(solutions):
    """
    Checks for the amount of Tas that are assigned to sections in which they marked "unwilling"
    Param: solutions: numpy array, one solution
    Return: int, number of sections unwilling tas are assigned to for that sol
    """
    
    # marks every unwilling section with a 1, marks all other sections with 0
    unwilling_lst = np.where((solutions == 1) & (section_prefs == 'U'), 1, 0)

    # adds 1 to array for each unwillingly assigned sections
    unwilling_count = [sum(lst) for lst in unwilling_lst]

    # returns total number of unwillingly assigned sections
    return sum(unwilling_count)


def unpreferred(solutions):
    """
    Counts the amount of Tas that were assigned to sections in which their preference was "unpreferred"
    Param: numpy array, one solution
    Return: int, number of sections unpreferred tas are assigned to for that sol
    """

    # marks every unpreferred section with a 1, marks all other sections with 0
    willing_lst = np.where((solutions == 1) & (section_prefs == 'W'), 1, 0)

    # adds 1 to array for each unpreferred but assigned sections
    willing_count = [sum(lst) for lst in willing_lst]

    # returns total number of unpreferred but assigned sections
    return sum(willing_count)


def swapper(solutions):
    """
    Swaps two random rows
    Param: solutions: numpy array, one solution
    Return: new solution generated from original
    """
    # accesses single solution
    new = solutions[0]

    # chooses two random rows within solutions
    i = rnd.randrange(0, len(new))
    j = rnd.randrange(0, len(new))

    # swaps random rows
    new[i], new[j] = new[j], new[i]

    # returns new solution
    return new


def reallocate(solutions):
    """
    Finding which Ta's are overallocated, and swapping the index of an assigned section to unassigned
    Param: solutions: numpy array, one solution
    Returns: update solutions
    """

    # accesses single solution
    new = solutions[0]

    # list of position in sol of each ta who is overallocated
    over = [i for ta, max, i in zip(new, allo_prefs, range(len(new))) if sum(ta) > max]

    # if no tas overallocated
    if not over:
        return new

    # choose random overallocated ta
    ta = rnd.choice(over)

    while True:
        i = rnd.randrange(0, len(new[ta]))

        # if section assigned, unassign
        if new[ta][i] == 1:
            new[ta][i] = 0
            return new


def trade_rows(solutions):
    """
    Swaps one row of a solution with another solution's row
    Param: solutions: numpy array, one solution
    Return: new solution generated from original
    """
    # accesses first solution
    sol1 = solutions[0]

    # accesses second solution
    sol2 = solutions[1]

    # chooses random row i
    i = rnd.randrange(0, len(sol1))

    # swaps row i of sol 1 with row i of sol 2
    sol1[i] = sol2[i]
    return sol1


def lessen_unw(solutions):
    """
    Unassigns all TAs that are marked to a section they marked as unwilling from the solution
    Param: solutions: numpy array, one solution
    Return: new solution generated from original
    """
    # accesses single solution
    new = solutions[0]

    # get array of sections assigned where unwilling
    unwilling = np.where((new == 1) & (section_prefs == 'U'), 1, 0)

    # positions of tas that are assigned to unwilling sections
    tas_unwilling = [i for assigned, i in zip(unwilling, range(len(unwilling))) if 1 in assigned]

    if not tas_unwilling:
        return new

    ta = rnd.choice(tas_unwilling)

    # unassign all unwilling sections for random ta
    new[ta] = [0 if unwilling[ta][i] == 1 else new[ta][i] for i in range(len(new[ta]))]

    return new


def swap_will(solutions):
    """
    Takes one TA who is assigned to an unwilling section and assigns them to a section they marked as "willing"
    Param: solutions: numpy array, one solution
    Return: new solution generated from original
    """
    new = solutions[0]

    # array of sections assigned where unwilling and unassigned where willing
    unwilling = np.where((new == 1) & (section_prefs == 'U'), 1, 0)
    willing = np.where((new == 0) & (section_prefs != 'U'), 1, 0)

    # positions of tas that are assigned to unwilling sections
    tas_unwilling = [i for assigned, i in zip(unwilling, range(len(unwilling))) if 1 in assigned]

    if not tas_unwilling:
        return new

    ta = rnd.choice(tas_unwilling)

    # get random unwilling/assigned section and willing/unassigned section for random ta
    unassign = rnd.choice([i for assigned, i in enumerate(unwilling[ta]) if assigned == 1])
    assign = rnd.choice([i for unassigned, i in enumerate(willing[ta]) if unassigned == 1])

    # unassign unwilling, assign willing
    new[ta][unassign], new[ta][assign] = 0, 1

    return new


def min_under(solutions):
    """
    Agent that checks for understaffed TA sections, and allocates a random TA to the section
    Param: solutions: numpy array, one solution
    Return: new solution generated from original
    """
    new = solutions[0]

    # positions of sections that are undersupported
    under = [i for section, min, i in zip(new.T, sections['min_ta'].values, range(len(new.T))) if sum(section) < min]

    # If list is empty, no changes are made
    if not under:
        return new

    section = rnd.choice(under)
    ta = rnd.randrange(0, len(new))

    # assign random ta to random undersupported section
    new[ta][section] = 1

    return new


def change_assigned(solutions):
    """
    Picks a random ta and section and swaps the assignment (0 to 1 or 1 to 0)
    Param: solutions: numpy array, one solution
    Return: new solution generated from original
    """
    new = solutions[0]
    i = rnd.randrange(0, len(new))
    j = rnd.randrange(0, len(new[0]))

    new[i][j] = (new[i][j] + 1) % 2
    return new


def main():
    preferences = pd.read_csv('sections_easy.csv')

    # Create framework
    E = Evo()

    # Register objectives
    E.add_fitness_criteria("overallocation", overallocation)
    E.add_fitness_criteria("conflicts", conflicts)
    E.add_fitness_criteria("undersupport", undersupport)
    E.add_fitness_criteria("unwilling", unwilling)
    E.add_fitness_criteria("unpreferred", unpreferred)
    E.add_fitness_criteria("no_tas", no_tas)

    # Register some agents
    E.add_agent("swapper", swapper, k=1)
    E.add_agent("trader", trade_rows, k=2)
    E.add_agent("eliminate_unwilling", swap_will, k=1)
    E.add_agent("reallocate", reallocate, k=1)
    E.add_agent("change_assigned", change_assigned)
    E.add_agent("min_under", min_under)
    E.add_agent("lessen_unw", lessen_unw)

    # Seed the population with an initial random solution
    L = np.array([[rnd.choice([0, 0, 0, 0, 0, 1]) for _ in range(17)] for _ in range(43)])
    E.add_solution(L)
    print(E)


    # Run the evolver
    E.evolve(100000, 200, 1000, 600)

    # Print final results
    print(E)


if __name__ == '__main__':
    main()
