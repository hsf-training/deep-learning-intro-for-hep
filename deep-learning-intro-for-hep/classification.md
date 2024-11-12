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

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
penguins_df = pd.read_csv("data/penguins.csv")
penguins_df
```

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

+++

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
