"""Smoke check, small enough for CI.

- the test set keeps every row (outliers are only removed from train);
- mutation never changes the parent's layout;
- a tiny GA run evaluates on validation and records every generation.
"""
import os

os.environ.setdefault("GA_EPOCHS", "2")

from EDA import *
from Mutation import mutate_individual
from Pool import Pool

# 7032 clean rows: test is 20% of them, validation 25% of the rest, both untouched.
assert len(y_test) == 1407, len(y_test)
assert len(y_val) == 1407, len(y_val)

pool = Pool(size=4, optim="max", x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
parent = pool.individuals[0]
before = [vars(layer).copy() for layer in parent.layer_layout]
for _ in range(20):
    mutate_individual(parent, mutation_probability=1.0)
assert [vars(layer) for layer in parent.layer_layout] == before, "mutation changed the parent"

best, best_curve, mean_curve, _ = pool.evolve(True, False, generations=2)
assert 0.0 <= best <= 1.0
assert len(best_curve) == len(mean_curve) == 3, best_curve
print(f"best validation macro F1 after 2 generations: {best:.3f}")
print("all checks passed")
