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

# Classification in PyTorch

+++

This is a continuation of the previous section, introducing PyTorch for two basic problems: regression and classification.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Penguins again!

+++

See [Regression in PyTorch](regression.md) for instructions on how to get the data.

```{code-cell} ipython3
penguins_df = pd.read_csv("data/penguins.csv")
penguins_df
```

This time, we're going to focus on the categorical variables, `species`, `island`, `sex`, and `year`.

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

Categorical problems involve at least one categorical variable. For instance, given a penguin's bill length and bill depth, what's its species? We might ask a categorical model to predict the most likely category or we might ask it to tell us the probabilities of each category.

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

It's also possible for all of the features and predictions in a problem to be categorical. For instance, suppose the model is given a penguin's species and is asked to predict its probable island (so we can bring it home!).

```{code-cell} ipython3
pd.get_dummies(penguins_df.dropna()[["species", "island"]])
```

Here's what we'll use as our sample problem: given the bill length and depth, what is the penguin's species?

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
    ax.set_xlabel("bill length (mm)")
    ax.set_ylabel("bill depth (mm)")

    ax.legend(loc="lower left", framealpha=1)

plot_categorical_problem(ax)

plt.show()
```

The model will be numerical, a function from bill length and depth to a 3-dimensional probability space. Probabilities have two hard constraints:

* they are all strictly bounded between $0$ and $1$
* all the probabilities in a set of possibilities need to add up to $1$.

If we define $P_A$, $P_G$, and $P_C$ for the probability that a penguin is Adelie, Gentoo, or Chinstrap, respectively, then $P_A + P_G + P_C = 1$ and all are non-negative.

One way to ensure the first constraint is to let a model predict values between $-\infty$ and $\infty$, then pass them through a sigmoid function:

$$p(x) = \frac{1}{1 + \exp(x)}$$

If we only had 2 categories, $P_1$ and $P_2$, this would be sufficient: we'd model the probability of $P_1$ only by applying a sigmoid as the last step in our model. $P_2$ can be inferred from $P_1$.

But what if we have 3 categories, as in the penguin problem?

The sigmoid function has a multidimensional generalization called [softmax](https://en.wikipedia.org/wiki/Softmax_function). Given an $n$-dimensional vector $\vec{x}$ with components $x_1, x_2, \ldots, x_n$,

$$P(\vec{x})_i = \frac{\exp(x_i)}{\displaystyle \sum_j^n \exp(x_j)}$$

For any $x_i$ between $-\infty$ and $\infty$, all $0 \le P_i \le 1$ and $\sum_i P_i = 1$. Thus, we can pass the output of any $n$-dimensional vector through a softmax to get probabilities.

+++

## Scikit-Learn

+++

In keeping with the principle that a linear fit is the simplest kind of neural network, we can use Scikit-Learn's `LogisticRegression` as a single-layer neural network.

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
```

Scikit-Learn requires the target classes to be passed as integer labels, so we use `categorical_int_df`, rather than `categorical_1hot_df`.

```{code-cell} ipython3
categorical_features = categorical_int_df[["bill_length_mm", "bill_depth_mm"]].values
categorical_targets = categorical_int_df["species"].values
```

The reason we set `penalty=None` is (again) to stop it from applying regularization, which we'll learn about later.

```{code-cell} ipython3
best_fit = LogisticRegression(penalty=None).fit(categorical_features, categorical_targets)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

def plot_categorical_solution(ax, model, xlow=29, xhigh=61, ylow=12, yhigh=22):
    # compute the three probabilities for every 2D point in the background
    background_x, background_y = np.meshgrid(np.linspace(xlow, xhigh, 100), np.linspace(ylow, yhigh, 100))
    background_2d = np.column_stack([background_x.ravel(), background_y.ravel()])
    probabilities = model(background_2d)
    
    # draw contour lines where the probabilities cross the 50% threshold
    ax.contour(background_x, background_y, probabilities[:, 0].reshape(background_x.shape), [0.5])
    ax.contour(background_x, background_y, probabilities[:, 1].reshape(background_x.shape), [0.5])
    ax.contour(background_x, background_y, probabilities[:, 2].reshape(background_x.shape), [0.5])

plot_categorical_solution(ax, lambda x: best_fit.predict_proba(x))
plot_categorical_problem(ax)

plt.show()
```

Despite being called a linear model, the best fit lines are curvy. That's because it's a linear fit from the 2-dimensional input space to a 3-dimensional space, which ranges from $-\infty$ to $\infty$ in all 3 dimensions, and then those are passed through a softmax to get 3 well-behaved probabilities.

To add a layer of ReLU functions, use Scikit-Learn's [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

```{code-cell} ipython3
from sklearn.neural_network import MLPClassifier
```

```{code-cell} ipython3

```
