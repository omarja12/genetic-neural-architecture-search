"""Evolve neural-network architectures for churn prediction, with and without
elitism, then score the best architecture once on the held-out test set.

GA_GENERATIONS, GA_SIZE and GA_EPOCHS override the defaults in EDA.py.
"""
from EDA import *
from LayerLayout import *
from Individual import *
from Pool import *
from Selections import *


def genetic_algorithm_plot(kwargs):
    """
    Evolve two copies of the same starting pool, one without and one with
    elitism, plot best and mean validation fitness per generation, and
    return the two final pools.
    """
    pop1 = Pool(**kwargs)
    pop2 = deepcopy(pop1)
    tmp1 = pop1.evolve(False, False)
    tmp2 = pop2.evolve(True, False)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(range(1, len(tmp1[1])+1), tmp1[1], '-g', label='best fitness')
    ax[0].plot(range(1, len(tmp1[2])+1), tmp1[2], '-b', label='mean fitness')

    ax[1].plot(range(1, len(tmp2[1])+1), tmp2[1], '-g', label='best fitness')
    ax[1].plot(range(1, len(tmp2[2])+1), tmp2[2], '-b', label='mean fitness')

    for axis, title in zip(ax, ["Without elitism", "With elitism"]):
        axis.grid(axis='both', linestyle='--', color="#add8e6")
        axis.legend(loc="best")
        axis.set_ylabel('Validation macro F1')
        axis.set_title(title)
    ax[1].set_xlabel('Generation')

    fig.tight_layout()
    fig.savefig("fitness_by_generation.png", dpi=120)
    plt.show()
    return pop1, pop2


if __name__ == "__main__":
    d = {"size": SIZE,
         "optim": OPTIM,
         "x_train": x_train,
         "y_train": y_train,
         "x_val": x_val,
         "y_val": y_val}

    pools = genetic_algorithm_plot(d)
    best = max((ind for pool in pools for ind in pool.individuals), key=attrgetter("fitness"))
    print(f"Best architecture: {best}")
    print(f"Validation macro F1: {best.fitness:.3f}")
    print(f"Test macro F1:       {best.score(x_test, y_test):.3f}")
