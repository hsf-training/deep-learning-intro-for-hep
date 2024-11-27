---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Hyperparameters and validation

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
```

## Some definitions

+++

As we've seen, the numbers in a model that a minimization algorithm optimizes are called "**parameters**" (or "weights"):

```{code-cell} ipython3
model = nn.Sequential(
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
)
```

```{code-cell} ipython3
list(model.parameters())
```

I've discussed other changeable aspects of models,

* architecture (number of hidden layers, how many nodes each, maybe other graph structures),
* choice of activation function,
* regularization techniques,
* choice of input features,

as well as other changeable aspects of the training procedure,

* minimization algorithm and its options (such as learning rate and momentum),
* distribution of initial parameter values in each layer,
* number of epochs and mini-batch size.

It would be confusing to call these choices "parameters," so we call them "**hyperparameters**" ("hyper" means "over or above"). The problem of finding the best model is split between you, the human, choosing hyperparameters and the optimizer choosing parameters. Following the farming analogy from the [Overview](01-overview.md), the hyperparameters are choices that the farmer gets to make—how much water, how much sun, etc. The parameters are the low-level aspects of how a plant grows, where its leaves branch, how its veins and roots organize themselves to survive. Generally, there are a lot more parameters than hyperparameters.

If there are a lot of hyperparameters to tune, we might want to tune them algorithmically—maybe with a grid search, randomly, or with Bayesian optimization. Technically, I suppose they become parameters, or we get a three-level hierarchy: parameters, hyperparameters, and hyperhyperparameters! Practitioners might not use consistent terminology ("using ML to tune hyperparameters" is a contradiction in terms), but just don't get confused about who is optimizing which: algorithm 1, algorithm 2, or human. Even if some hyperparameters are being tuned by an algorithm, some of them must be chosen by hand. For instance, you choose a type of ML algorithm, maybe a neural network, maybe something else, and non-numerical choices about the network topology are generally hand-chosen. If a grid search, random search, or Bayesian optimization is choosing the rest, you do have to set the grid spacing for the grid search, the number of trials and measure of the random search, or various options in the Bayesian search. Or, a software package that you use chooses for you.

+++

## Partitioning data into training, validation, and test samples

+++

In the section on [Regularization](16-regularization.md), we split a dataset into two samples and computed the loss function on each.

* **Training:** loss computed from the training dataset is used to change the parameters of the model. Training loss can get arbitrarily small as the model is adjusted to fit the training data points exactly (if it has enough parameters to be so flexible).
* **Test:** loss computed from the test dataset acts as an independent measure of the model quality. A model generalizes well if it is a good fit (has minimal loss) on both the training data and data drawn from the same distribution: the test dataset.

Suppose that I set up an ML model with some hand-chosen hyperparameters, optimize it for the training dataset, and then I don't like how it performs on the test dataset, so I adjust the hyperparameters and run again. And again. After many hyperparameter adjustments, I find a set that optimizes both the training and the test datasets. Is the test dataset an independent measure of the model quality?

It's not a fair test because my hyperparameter optimization is not a fundamentally different thing from the automated parameter optimization. When I adjust hyperparameters, look at how the loss changes, and use that information to either revert the hyperparameters or make another change, I am acting as a minimization algorithm—just a slow, low-dimensional one.

Since we do need to optimize (some of) the hyperparameters, we need a third data subsample:

* **Validation:** loss computed from the validation dataset is used to change the hyperparameters of the model.

So we need to do a 3-way split of the original dataset. A common practice is to use 80% of the data for training, 10% of the data for validation, and hold 10% of the data for the final test—do not look at its loss value until you're sure you won't be changing hyperparameters anymore. This is similar to the practice, in particle physics, of performing a blinded analysis: you can't look at the analysis result until you are no longer changing the analysis procedure (and then you're stuck with it).

The fractions, 80%, 10%, 10%, are conventional. They're not hyperparameters—you can't change the proportions during model-tuning. Since the validation and test datasets are smallest, their sizes set the resolution of the model evaluation, so you might need to increase them (to, say, 60%, 20%, 20%) if you know that 10% of your data won't be enough to quantify precision. But if you're statistics-limited, neural networks might not be the best ML model (consider Boosted Decision Trees (BDTs) instead).

PyTorch's [random_split](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split) function can split a dataset into 3 parts as easily as 2.

```{code-cell} ipython3
boston_prices_df = pd.read_csv(
    "data/boston-house-prices.csv", sep="\s+", header=None,
    names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"],
)
boston_prices_df = (boston_prices_df - boston_prices_df.mean()) / boston_prices_df.std()

features = boston_prices_df.drop(columns=["MEDV"])
targets = boston_prices_df["MEDV"]
```

```{code-cell} ipython3
from torch.utils.data import TensorDataset, DataLoader, random_split
```

```{code-cell} ipython3
features_tensor = torch.tensor(features.values, dtype=torch.float32)
targets_tensor = torch.tensor(targets.values[:, np.newaxis], dtype=torch.float32)

dataset = TensorDataset(features_tensor, targets_tensor)

train_size = int(np.floor(0.8 * len(dataset)))
valid_size = int(np.floor(0.1 * len(dataset)))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

len(train_dataset), len(valid_dataset), len(test_dataset)
```

Oddly, Scikit-Learn's equivalent, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), can only return 2 parts. If you use it, you have to use it like this:

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
```

```{code-cell} ipython3
train_features, tmp_features, train_targets, tmp_targets = train_test_split(features.values, targets.values, train_size=0.8)
valid_features, test_features, valid_targets, test_targets = train_test_split(tmp_features, tmp_targets, train_size=0.5)

del tmp_features, tmp_targets

len(train_features), len(valid_features), len(test_features)
```

although Scikit-Learn does return NumPy arrays or Pandas DataFrames, which are more useful than PyTorch Datasets if you can fit everything into memory.

+++

## Cross-validation

+++

For completeness, I should mention an alternative to allocating a validation dataset: you can cross-validate on a larger subsample of the data. In this method, you still need to isolate a test sample for final evaluation, but you can optimize the parameters and hyperparameters using the same data. The following diagram from [Scikit-Learn's documentation](https://scikit-learn.org/stable/modules/cross_validation.html) illustrates it well:

![](img/grid_search_cross_validation.png){. width="75%"}

After isolating a test sample (maybe 20%), you

1. subdivide the remaining sample into $k$ subsamples,
2. for each $i \in [0, k)$, combine all data except for subsample $i$ into a training dataset $T_i$ and use subsample $i$ as a validation dataset $V_i$,
3. train an independent model on each $T_i$ and compute the validation loss $L_i$ with the corresponding trained model and validation dataset $V_i$,
4. the total validation loss is $L = \sum_i L_i$.

This is more computationally expensive, but it makes better use of smaller datasets.

Scikit-Learn provides a [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) object to help keep track of indexes when cross-validating. For $k = 5$,

```{code-cell} ipython3
from sklearn.model_selection import KFold
```

```{code-cell} ipython3
kf = KFold(n_splits=5, shuffle=True)
```

By calling [KFold.split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold.split) on a dataset with a length (that is, an object you can call `len` on to get its length), you can iterate over folds ($i \in [0, k)$) and get random subsamples $T_i$ and $V_i$ as arrays of integer indexes.

```{code-cell} ipython3
for train_indexes, valid_indexes in kf.split(dataset):
    print(len(train_indexes), len(valid_indexes))
```

```{code-cell} ipython3
train_indexes[:20]
```

```{code-cell} ipython3
valid_indexes[:20]
```

These integer indexes can slice arrays, Pandas DataFrames (via [pd.DataFrame.iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)), and PyTorch Tensors.

```{code-cell} ipython3
train_features_i, train_targets_i = dataset[train_indexes]
valid_features_i, valid_targets_i = dataset[valid_indexes]
```

Here's a full example that computes training loss and validation loss, using cross-validation from the Boston House Prices dataset.

```{code-cell} ipython3
NUMBER_OF_FOLDS = 5
NUMBER_OF_EPOCHS = 1000

kf = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=True)

# use a class so that we can generate new, independent models for every k-fold
class Model(nn.Module):
    def __init__(self):
        super().__init__()   # let PyTorch do its initialization first
        self.model = nn.Sequential(
                nn.Linear(13, 100),
                nn.ReLU(),
                nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.model(x)

# initialize loss-versus-epoch lists as zeros to update with each k-fold
train_loss_vs_epoch = [0] * NUMBER_OF_EPOCHS
valid_loss_vs_epoch = [0] * NUMBER_OF_EPOCHS

# for each k-fold
for train_indexes, valid_indexes in kf.split(dataset):
    train_features_i, train_targets_i = dataset[train_indexes]
    valid_features_i, valid_targets_i = dataset[valid_indexes]

    # generate a new, independent model, loss_function, and optimizer
    model = Model()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # do a complete training loop
    for epoch in range(NUMBER_OF_EPOCHS):
        optimizer.zero_grad()

        train_loss = loss_function(model(train_features_i), train_targets_i)
        valid_loss = loss_function(model(valid_features_i), valid_targets_i)

        train_loss.backward()
        optimizer.step()

        # average loss over k-folds (could ignore NUMBER_OF_FOLDS to sum, instead)
        train_loss_vs_epoch[epoch] += train_loss.item() / NUMBER_OF_FOLDS
        valid_loss_vs_epoch[epoch] += valid_loss.item() / NUMBER_OF_FOLDS
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(range(1, len(train_loss_vs_epoch) + 1), train_loss_vs_epoch, label="training k-folds")
ax.plot(range(1, len(valid_loss_vs_epoch) + 1), valid_loss_vs_epoch, color="tab:blue", ls=":", label="validation k-folds")

ax.set_ylim(0, min(max(train_loss_vs_epoch), max(valid_loss_vs_epoch)))
ax.set_xlabel("epoch number")
ax.set_ylabel("loss")

ax.legend(loc="upper right")

plt.show()
```

But since you'll usually have large datasets (usually Monte Carlo, in HEP), you can usually just split the data 3 ways between training, validation, and test datasets, without mixing training and validation using cross-validation.
