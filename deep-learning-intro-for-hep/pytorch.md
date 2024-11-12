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

# Classification and regression in PyTorch

+++

This section introduces PyTorch so that we can use it for the remainder of the course. Whereas Scikit-Learn gives you a function for just about [every type of machine learning model](https://scikit-learn.org/stable/machine_learning_map.html), PyTorch gives you the pieces and expects you to build it yourself. (The [JAX](https://jax.readthedocs.io/) library is even more extreme in providing only the fundamental pieces. PyTorch's level of abstraction is between JAX and Scikit-Learn.)

I'll use the two types of problems we've seen so far—regression and classification—to show Scikit-Learn and PyTorch side-by-side. First, though, let's get a dataset that will provide us with realistic regression and classification problems.

+++

## Penguins!

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

This is my new favorite dataset: basic measurements of 3 species of penguins. You can get the data as a CSV file from the [original source](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris) or from this project's GitHub: [deep-learning-intro-for-hep/data/penguins.csv](https://github.com/hsf-training/deep-learning-intro-for-hep/blob/main/deep-learning-intro-for-hep/data/penguins.csv).

![](img/culmen_depth.png){. width="50%"}

Replace `data/penguins.csv` with the file path where you saved the file after downloading it.

```{code-cell} ipython3
penguins_df = pd.read_csv("data/penguins.csv")
penguins_df
```

This dataset has numerical features, such as `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` and `year` of data-taking, and it has categorical features like `species`, `island`, and `sex`. Some of the measurements are missing (`NaN`), but we'll ignore them with [pd.DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html).

+++

## Numerical and categorical data

+++

Numerical versus categorical may be thought of as data types in a programming language: integers and floating-point types are numerical, booleans and strings are categorical. However, they can also be thought of as the [fundamental types in data analysis](https://en.wikipedia.org/wiki/Level_of_measurement), which determines which set of mathematical operations are meaningful:

| Level | Math | Description | Physics example |
|:--|:--:|:--|:--|
| Nominal category | =, ≠ | categories without order | jet classification, data versus Monte Carlo |
| Ordinal category | >, < | categories that have an order | barrel region, overlap region, endcap region |
| Interval number | +, ‒ | doesn't have an origin | energy, voltage, position, momentum |
| Ratio number | ×, / | has an origin | absolute temperature, mass, opening angle |

Python's `TypeError` won't tell you if you're inappropriately multiplying or dividing an interval number, but it will tell you if you try to subtract strings.

**Regression** problems are ones in which the features and the predictions are both numerical (interval numbers, at least). For instance, given a penguin's flipper length, what's its mass?

```{code-cell} ipython3
regression_features, regression_targets = penguins_df.dropna()[["flipper_length_mm", "body_mass_g"]].values.T
```

```{code-cell} ipython3
fig, ax = plt.subplots()

def plot_regression_problem(ax, xlow=170, xhigh=235, ylow=2400, yhigh=6500):
    ax.scatter(regression_features, regression_targets, marker=".")
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel("flipper length (mm)")
    ax.set_ylabel("body mass (g)")

plot_regression_problem(ax)

plt.show()
```

**Categorical** problems involve at least one categorical variable. For instance, given a penguin's bill length and bill depth, what's its species? We might ask a categorical model to predict the most likely category or we might ask it to tell us the probabilities of each category.

We can't pass string-valued variables into a fitter, so we need to convert the strings into numbers. Since these categories are nominal (not ordinal), equality/inequality is the only meaningful operation, so the numbers should only indicate which strings are the same as each other and which are different.

Pandas has a function, [pd.factorize](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html), to turn unique categories into unique integers and an index to get the original strings back. (You can also use Pandas's [categorical dtype](https://pandas.pydata.org/docs/user_guide/categorical.html).)

```{code-cell} ipython3
categorical_int_df = penguins_df.dropna()[["bill_length_mm", "bill_depth_mm", "species"]]
categorical_int_df["species"], code_to_name = pd.factorize(categorical_int_df["species"].values)
categorical_int_df
```

```{code-cell} ipython3
code_to_name
```

This is called an "integer encoding" or "label encoding."

Pandas also has a function, [pd.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), that turns $n$ unique categories into an $n$-dimensional space of booleans. As training data, these represent the probabilities of each species: `False` is $0$ and `True` is $1$ and the _given_ classifications are certain.

```{code-cell} ipython3
categorical_1hot_df = pd.get_dummies(penguins_df.dropna()[["bill_length_mm", "bill_depth_mm", "species"]])
categorical_1hot_df
```

This is called [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) and it's generally more useful than integer encoding, though it takes more memory (especially if you have a lot of distinct categories).

For instance, suppose that the categorical variable is the feature and we're trying to predict something numerical:

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.scatter(categorical_int_df["species"], categorical_int_df["bill_length_mm"])
ax.set_xlabel("species")
ax.set_ylabel("bill length (mm)")

plt.show()
```

If you were to fit a straight line through $x = 0$ and $x = 1$, it would have _some_ meaning: the intersections would be the average bill lengths of Adelie and Gentoo penguins, respectively. But if the fit also includes $x = 2$, it would be meaningless, since it would be using the order of Adelie, Gentoo, and Chinstrap, as well as the equal spacing between them, as relevant for determining the $y$ predictions.

On the other hand, the one-hot encoding is difficult to visualize, but any fits through this high-dimensional space are meaningful.

```{code-cell} ipython3
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")

jitter = np.random.normal(0, 0.05, (len(categorical_1hot_df), 3))

vis = ax.scatter(
    categorical_1hot_df["species_Adelie"] + jitter[:, 0],
    categorical_1hot_df["species_Gentoo"] + jitter[:, 1],
    categorical_1hot_df["species_Chinstrap"] + jitter[:, 2],
    c=categorical_1hot_df["bill_length_mm"],   # color of points is bill length
    s=categorical_1hot_df["bill_depth_mm"],    # size of points is bill depth
)
ax.set_xlabel("species is Adelie")
ax.set_ylabel("species is Gentoo")
ax.set_zlabel("species is Chinstrap")

plt.colorbar(vis, ax=ax, label="bill length (mm)", location="top")

plt.show()
```

It's also possible to consider problems in which all features and all predictions being categorical.

```{code-cell} ipython3
pd.get_dummies(penguins_df.dropna()[["species", "island"]])
```

We don't encounter these kinds of problems very often in HEP, though.

Here's what we'll use as our sample problem: given the bill length and depth, what is the penguin's species? The fact that there's more than 2 categories will require special handling.

```{code-cell} ipython3
fig, ax = plt.subplots()

def plot_categorical_problem(ax, xlow=29, xhigh=61, ylow=12, yhigh=22):
    df_Adelie = categorical_1hot_df[categorical_1hot_df["species_Adelie"] == 1]
    df_Gentoo = categorical_1hot_df[categorical_1hot_df["species_Gentoo"] == 1]
    df_Chinstrap = categorical_1hot_df[categorical_1hot_df["species_Chinstrap"] == 1]

    ax.scatter(df_Adelie["bill_length_mm"], df_Adelie["bill_depth_mm"], color="tab:blue", label="Adelie")
    ax.scatter(df_Gentoo["bill_length_mm"], df_Gentoo["bill_depth_mm"], color="tab:orange", label="Gentoo")
    ax.scatter(df_Chinstrap["bill_length_mm"], df_Chinstrap["bill_depth_mm"], color="tab:green", label="Chinstrap")

    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.legend(loc="lower left")

plot_categorical_problem(ax)

plt.show()
```

## Scikit-Learn and PyTorch for regression

+++

In keeping with the principle that a linear fit is the simplest kind of neural network, we can use Scikit-Learn's `LinearRegression` as a single-layer, no-activation neural network:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
```

```{code-cell} ipython3
best_fit = LinearRegression().fit(regression_features[:, np.newaxis], regression_targets)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

def plot_regression_solution(ax, model, xlow=170, xhigh=235):
    model_x = np.linspace(xlow, xhigh, 1000)
    model_y = model(model_x)
    ax.plot(model_x, model_y, color="tab:orange")

plot_regression_solution(ax, lambda x: best_fit.predict(x[:, np.newaxis]))
plot_regression_problem(ax)

plt.show()
```

Next, let's add a layer of ReLU functions using Scikit-Learn's `MLPRegressor`. The reason we set `alpha=0` is because its regularization is not off by default, and we haven't talked about regularization yet. The `solver="lbfgs"` picks a more robust optimization method for this low-dimension problem.

```{code-cell} ipython3
from sklearn.neural_network import MLPRegressor
```

```{code-cell} ipython3
best_fit = MLPRegressor(
    activation="relu", hidden_layer_sizes=(5,), solver="lbfgs", max_iter=1000, alpha=0, random_state=123
).fit(regression_features[:, np.newaxis], regression_targets)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

plot_regression_solution(ax, lambda x: best_fit.predict(x[:, np.newaxis]))
plot_regression_problem(ax)

plt.show()
```

Now let's do the same in PyTorch. First, the linear model: `nn.Linear(1, 1)` means a linear transformation from a 1-dimensional space to a 1-dimensional space.

```{code-cell} ipython3
import torch
import torch.nn as nn
import torch.optim as optim
```

```{code-cell} ipython3
model = nn.Linear(1, 1)
model
```

A model has parameters that PyTorch will vary in the fit. When you create a model, they're already given random values (one slope and one intercept, in this case). `requires_grad` refers to the fact that the derivatives of the parameters are also tracked, for the optimization methods that use derivatives.

```{code-cell} ipython3
list(model.parameters())
```

We can't pass NumPy arrays directly into PyTorch—they have to be converted into PyTorch's own array type (which can reside on CPU or GPU), called `Tensor`.

PyTorch's functions are very sensitive to the exact data types of these tensors: the difference between integers and floating-point can make PyTorch run a different algorithm! For floating-point numbers, PyTorch prefers 32-bit.

```{code-cell} ipython3
tensor_features = torch.tensor(regression_features[:, np.newaxis], dtype=torch.float32)
tensor_targets = torch.tensor(regression_targets[:, np.newaxis], dtype=torch.float32)
```

Now we need to say _how_ we're going to train the model.

* What will the loss function be? For a regression problem, it would usually be $\chi^2$, or mean squared error: [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
* Which optimizer should we choose? (This is the equivalent of `solver="lbfgs"` in Scikit-Learn.) We'll talk more about these later, and the right choice will usually be [nn.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam), but not for this linear problem. For now, we'll use [nn.Rprop](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop).

The optimizer has access to the model's parameters, and it will modify them in-place.

```{code-cell} ipython3
loss_function = nn.MSELoss()

optimizer = optim.Rprop(model.parameters())
```

To actually train the model, you have to write your own loop! It's more verbose, but you get to control what happens and debug it.

One step in optimization is called an "epoch." In Scikit-Learn, we set `max_iter=1000` to get 1000 epochs.

```{code-cell} ipython3
for epoch in range(1000):
    # tell the optimizer to begin an optimization step
    optimizer.zero_grad()

    # use the model as a prediction function: features → prediction
    predictions = model(tensor_features)

    # compute the loss (χ²) between these predictions and the intended targets
    loss = loss_function(predictions, tensor_targets)

    # tell the loss function and optimizer to end an optimization step
    loss.backward()
    optimizer.step()
```

The `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` calls change the state of the optimizer and the model parameters, but you can think of them just as the beginning and end of an optimization step.

There are other state-changing functions, like `model.train()` (to tell it we're going to start training) and `model.eval()` (to tell it we're going to start using it for inference), but we won't be using any of the features that depend on the variables that these set.

Now, to draw a plot with this model, we'll have to turn the NumPy `x` positions into a `Tensor`, run it through the model, and then convert the model's output back into a NumPy array. The output has derivatives as well as values, so those will need to be detached.

* NumPy `x` to Torch: `torch.tensor(x, dtype=torch.float32)` (or other dtype)
* Torch `y` to NumPy: `y.detach().numpy()`

```{code-cell} ipython3
fig, ax = plt.subplots()

def numpy_model(x):
    tensor_x = torch.tensor(x[:, np.newaxis], dtype=torch.float32)
    return model(tensor_x).detach().numpy()

plot_regression_solution(ax, numpy_model)
plot_regression_problem(ax)

plt.show()
```

A layered neural network in PyTorch is usually represented by a class, such as this:

```{code-cell} ipython3
class NeuralNetworkWithReLU(nn.Module):
    def __init__(self, hidden_layer_size):
        super().__init__()   # let PyTorch do its initialization first

        self.step1 = nn.Linear(1, hidden_layer_size)  # 1D input → 5D
        self.step2 = nn.ReLU()                        # 5D ReLU
        self.step3 = nn.Linear(hidden_layer_size, 1)  # 5D → 1D output

    def forward(self, x):
        return self.step3(self.step2(self.step1(x)))

model = NeuralNetworkWithReLU(5)
model
```

```{code-cell} ipython3
list(model.parameters())
```

You can initialize it with as many sub-models as you want and then implement what they do to features `x` in the `forward` method.

However, I like [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) better for models that are simple sequences of layers.

```{code-cell} ipython3
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)
model
```

```{code-cell} ipython3
list(model.parameters())
```

Let's fit this one the same way we fit the single-layer model:

```{code-cell} ipython3
loss_function = nn.MSELoss()
optimizer = optim.Rprop(model.parameters())

for epoch in range(1000):
    # tell the optimizer to begin an optimization step
    optimizer.zero_grad()

    # use the model as a prediction function: features → prediction
    predictions = model(tensor_features)

    # compute the loss (χ²) between these predictions and the intended targets
    loss = loss_function(predictions, tensor_targets)

    # tell the loss function and optimizer to end an optimization step
    loss.backward()
    optimizer.step()
```

```{code-cell} ipython3
fig, ax = plt.subplots()

def numpy_model(x):
    tensor_x = torch.tensor(x[:, np.newaxis], dtype=torch.float32)
    return model(tensor_x).detach().numpy()

plot_regression_solution(ax, numpy_model)
plot_regression_problem(ax)

plt.show()
```

Chances are, you don't see any evidence of the ReLU and the above is just a straight line.

Scroll back up to the initial model parameters, and now look at them after the fit:

```{code-cell} ipython3
list(model.parameters())
```

Initially, the model parameters are all random numbers between $-1$ and $1$. After fitting, _some_ of the parameters are in the few-hundred range.

Now look at the $x$ and $y$ ranges on the plot: flipper lengths are hundreds of millimeters and body masses are thousands of grams. The optimizer had to gradually step values of order 1 up to values of order 100‒1000. The optimizer took small steps to avoid jumping over the solution. In the end, the optimizer found a reasonably good fit by scaling just a few parameters up and effectively performing a purely linear fit.

We should have scaled the inputs and outputs so that the values the fitter sees are _all_ of order 1. This is something that PyTorch _assumes_ you will do.

In many applications, I've seen people scale the data independently of the model. However, I'd like to make the scaling a part of the model. We could add a `nn.Linear(1, 1, bias=False)` to multiply by a parameter, but this would become a new parameter for the optimizer to tune in the fit. Instead, I'll use PyTorch's [nn.Module.register_buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) to add a fixed constant to the model (which it would save if it saves the model to a file).

```{code-cell} ipython3

```
