# Genetic Neural Architecture Search

A genetic algorithm that searches for a neural-network architecture to predict customer churn. Each individual is a Keras model: its number of hidden layers, layer types (dense or dropout), widths, activations, dropout rates, optimiser and loss are all genes. The GA trains each candidate, scores it, and breeds the next generation with tournament selection, layer-wise crossover and mutation, with or without elitism.

## How it works

| Module | Role |
|---|---|
| `EDA.py` | settings, data download, train / validation / test split, scaling and one-hot encoding |
| `LayerLayout.py` | one random hidden layer (the gene) |
| `Individual.py` | builds, trains and scores one architecture |
| `Selections.py` | tournament, ranking and fitness-proportionate selection |
| `Crossover.py` | swaps layer stacks at a random cut, and swaps optimiser and loss |
| `Mutation.py` | resamples optimiser, loss and depth on a copy of the parent |
| `Pool.py` | the population: evaluation, elitism and the generational loop |
| `Genetic_Programing_Neural_Networks.py` | runs the search with and without elitism and plots both |

**Evaluation.** The data is split 60 / 20 / 20 into train, validation and test, stratified on churn. Models train on train with early stopping on validation F1. GA fitness is macro F1 on validation. The test set is used once, to score the best architecture at the end. Outlier removal (Isolation Forest on tenure and charges) is fitted on train and only drops train rows.

## Run it

```bash
pip install -r requirements.txt
python check.py                                  # a few minutes on CPU
python Genetic_Programing_Neural_Networks.py     # full search: 20 generations of 12 models
```

For a quick run, set `GA_GENERATIONS`, `GA_SIZE` and `GA_EPOCHS`, for example `GA_GENERATIONS=3 GA_SIZE=6 GA_EPOCHS=10`. The script saves `fitness_by_generation.png` and prints the best architecture with its validation and test scores. The GitHub Actions workflow runs the check and a small search on every push and uploads the figure.

## Data

The [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) sample dataset from IBM (Apache-2.0): 7,043 customers, 20 features, 26.5% churn. It is downloaded to `data/` on first run.

## Notes

- `check.py` asserts that validation and test keep every row, that mutation never alters the parent, and that a short search records every generation.
- Fitness sharing is stubbed and raises `NotImplementedError`; run with it off.
- Dependencies are pinned for TensorFlow 2.15 on Python 3.11.
