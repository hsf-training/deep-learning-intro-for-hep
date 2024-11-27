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

# Exercise 4: Mini-batches and DataLoaders

+++

In this exercise, you'll make the loss versus epoch plots from the previous section, but using a PyTorch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

Here are the imports:

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
```

And here's the dataset:

```{code-cell} ipython3
boston_prices_df = pd.read_csv(
    "data/boston-house-prices.csv", sep="\s+", header=None,
    names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"],
)

# Pre-normalize the data so we can ignore that part of modeling
boston_prices_df = (boston_prices_df - boston_prices_df.mean()) / boston_prices_df.std()
```

```{code-cell} ipython3
features = torch.tensor(boston_prices_df.drop(columns="MEDV").values).float()
targets = torch.tensor(boston_prices_df["MEDV"]).float()[:, np.newaxis]
```

## The exercise

+++

See PyTorch's documentation on [Datasets and DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) to create a [TensorDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) from the `features` and `targets`, and then load that into two [DataLoaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), one with `batch_size=len(features)` (i.e. one big batch) and the other with `batch_size=50`. When complete, the `dataloader` can be used for iteration over mini-batches like this:

```python
for features_subset, targets_subset in dataloader:
    ...
```

You can use the following code to plot the loss versus epoch from the two batch sizes:

```python
fig, ax = plt.subplots()

ax.plot(range(1, len(loss_vs_epoch) + 1), loss_vs_epoch, label="one big batch")
ax.plot(range(1, len(loss_vs_epoch_batched) + 1), loss_vs_epoch_batched, label="mini-batches")

ax.set_xlabel("number of epochs")
ax.set_ylabel("loss ($\chi^2$)")

ax.legend(loc="upper right")

plt.show()
```

+++

## Something to think about

+++

Are you following this course because you have a particular ML problem in mind? Does its data fit in memory?

Note that you can make subclasses of the generic [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class like this:

```python
class CustomDataset(Dataset):
    def __len__(self):
        return np.iinfo(np.int64).max   # or number of batches, if known

    def __getitem__(self, batch_index):
        if can_get_more_data():
            features, targets = get_more_data()
            return features, targets
        else:
            raise IndexError("no more data")
```

For your particular dataset, how would you fit your data-loading procedure into the class above?
