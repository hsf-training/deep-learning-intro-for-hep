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

# Beyond supervised learning: clustering

+++

This course is about deep learning for particle physicists, but so far, its scope has been rather limited:

1. Our neural network examples haven't been particularly deep, and some of them were linear fits (depth = 0).
2. We've dealt with only two types of problems: supervised regression and supervised classification.
3. All of the neural networks we've seen have the simplest kind of topology: fully connected, non-recurrent.

The reason for (1) is that making networks deeper is just a matter of adding layers. Deep neural networks _used to be_ hard to train, but many of the technical considerations have been built into the tools we use: backpropagation, initial parameterization, ReLU, Adam, etc. Long computation times and hyperparameter tuning still present challenges, but you can only experience these issues in full-scale problems. This course focuses on understanding the pieces, because that's what you'll use to find your way out of such problems.

The reason for (2) is that many (not all) HEP problems are supervised. We usually know what functions we want to approximate because we have detailed Monte Carlo (MC) simulations of the fundamental physics and our detectors, so we can use "MC truth" as targets in training. This is very different from wanting ChatGPT to "respond like a human." Sometimes, however, particle physicists want algorithms to find unusual events or group points into clusters. These are unsupervised algorithms.

![](img/ml-for-everyone.jpg){. width="100%"}

(Diagram from [Machine Learning for Everyone](https://vas3k.com/blog/machine_learning/).)

For (3), there is no excuse. The fact that neural networks can be arbitrary graphs leaves wide open the question of which topology to use. That's where much of the research is. In the remainder of this course, we'll look at a few topologies, but I'll mostly leave you to explore the [neural network zoo](https://www.asimovinstitute.org/neural-network-zoo/) on your own.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## K-means clustering

+++

Clustering is a common task in HEP. In a clustering problem, we have a set of features $\vec{x}_i$ (for $i \in [0, N)$ and we want to find $n$ groups or "clusters" of the $N$ points such that $\vec{x}_i$ in the same cluster are close to each other and $\vec{x}_i$ in different clusters are far from each other.

If we know how many clusters we want to make, then the most common choice is k-means clustering. The k-means algorithm starts with $k$ initial cluster centers, $c_j$ (for $j \in [0, k)$) and labels all points in the space of $\vec{x}$ by the closest cluster: if $\vec{x}_i$ is closer to $c_j$ than all other $c_{j'}$ ($j' \ne j$), then $\vec{x}_i \in C_j$ where $C_j$ is the cluster associated with center $c_j$.

The algorithm then moves each cluster center $c_j$ to the mean $\vec{x}$ for all $\vec{x} \in C_j$. After enough iterations, the cluster centers gravitate to the densest accumulations of points. Note that this is _not_ a neural network. (We're getting to that.)

```{code-cell} ipython3
from sklearn.cluster import KMeans
```

In our penguin classification problem,

```{code-cell} ipython3
penguins_df = pd.read_csv("data/penguins.csv").dropna()
features = penguins_df[["bill_length_mm", "bill_depth_mm"]].values
hidden_truth = penguins_df["species"].values
```

we know there are 3 species of penguins, so let's try to find 3 clusters.

```{code-cell} ipython3
best_fit = KMeans(3).fit(features)
```

```{code-cell} ipython3
best_fit.cluster_centers_
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5))

def plot_clusters(best_fit, xlow=29, xhigh=61, ylow=12, yhigh=22):
    background_x, background_y = np.meshgrid(np.linspace(xlow, xhigh, 100), np.linspace(ylow, yhigh, 100))
    background_2d = np.column_stack([background_x.ravel(), background_y.ravel()])
    ax.contourf(background_x, background_y, best_fit.predict(background_2d).reshape(background_x.shape), alpha=0.2)

    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)

def plot_points(features, hidden_truth):
    ax.scatter(*features[hidden_truth == "Adelie"].T, label="actually Adelie")
    ax.scatter(*features[hidden_truth == "Gentoo"].T, label="actually Gentoo")
    ax.scatter(*features[hidden_truth == "Chinstrap"].T, label="actually Chinstrap")

    ax.set_xlabel("bill length (mm)")
    ax.set_ylabel("bill depth (mm)")

plot_clusters(best_fit)
plot_points(features, hidden_truth)

ax.scatter(*best_fit.cluster_centers_.T, color="white", marker="*", s=300)
ax.scatter(*best_fit.cluster_centers_.T, color="black", marker="*", s=100, label="cluster centers")

ax.legend(loc="lower left")

plt.show()
```

Remember that the k-means algorithm only sees the bill length and bill depth points _without_ species labels (without colors in the above plot). It uses the 3 cluster centers we gave it to split the data mostly vertically because the raw distribution is more clumpy in bill length than bill depth:

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.hist(features[:, 0], bins=20, range=(29, 61), histtype="step")
ax.set_xlabel("bill length (mm)")

for x, y in best_fit.cluster_centers_:
    label = ax.axvline(x, color="gray", ls="--")

ax.legend([label], ["cluster centers"], loc="upper left", framealpha=1)

plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.hist(features[:, 1], bins=20, range=(12, 22), histtype="step")
ax.set_xlabel("bill depth (mm)")

for x, y in best_fit.cluster_centers_:
    label = ax.axvline(y, color="gray", ls="--")

ax.legend([label], ["cluster centers"], loc="upper left", framealpha=1)

plt.show()
```

With more clusters, we can more identify more of its structure:

```{code-cell} ipython3
best_fit = KMeans(8).fit(features)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5))

plot_clusters(best_fit)
plot_points(features, hidden_truth)

ax.scatter(*best_fit.cluster_centers_.T, color="white", marker="*", s=300)
ax.scatter(*best_fit.cluster_centers_.T, color="black", marker="*", s=100, label="cluster centers")

ax.legend(loc="lower left")

plt.show()
```

But now we're subdividing clumps of data that we might want to keep together.

We can continue all the way from $k = 1$ (all points in a single cluster) to $k = n$ (every point in its own cluster) with this general trade-off: small $k$ can ignore structure and large $k$ can invent structure. There is a goodness-of-fit measure for this, [explained variance](https://en.wikipedia.org/wiki/Explained_variation), which compares the variance of cluster centers to the variance of points in each cluster: good clustering has small variance in each cluster. However, this can't be used to choose $k$, since it's optimal for $k = n$ (when every point is its own cluster, its variance is $0$).

+++

## Gaussian mixture models

+++

One of the assumptions built into k-means fitting is that points should belong to the cluster center that is closest to it in all dimensions equally. Thus, the area that belongs to each cluster is roughly circular (actually, [Voronoi tiles](https://en.wikipedia.org/wiki/Voronoi_diagram)). We can generalize the k-means algorithm a little bit by replacing each circularly symmetric cluster center with a Gaussian ellipsoid. Instead of a boolean membership like $\vec{x}_i \in C_j$, we can associate each $\vec{x}_i$ to all the clusters by varying degrees:

$$\mbox{membership}_{C_j}(\vec{x}_i) \propto \mbox{Gaussian}(\vec{x}_i; \vec{\mu}_j, \hat{\sigma}_j)$$

That is, a point $\vec{x}_i$ would _mostly_ belong to a cluster $C_j$ if it's within fewer standard deviations of the cluster's mean $\vec{\mu}_j$, scaled by its covariance matrix $\hat{\sigma}_j$, than other clusters $C_{j'}$ ($j' \ne j$). However, each point is a member of all clusters to different degrees, and some points may be on a boundary, where their membership to two clusters is about equal. We can turn this "soft clustering" into a "hard clustering" like k-means by considering only the maximum $\mbox{membership}_{C_j}(\vec{x}_i)$ for each point.

What's more important is that the covariance matrices allow the clusters to extend in long strips if necessary.

```{code-cell} ipython3
from sklearn.mixture import GaussianMixture
```

```{code-cell} ipython3
best_fit = GaussianMixture(3).fit(features)
```

This model has 3 mean vectors ($\vec{\mu}_j$):

```{code-cell} ipython3
best_fit.means_
```

and 3 covariance matrices ($\hat{\sigma}_j$):

```{code-cell} ipython3
best_fit.covariances_
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5))

plot_clusters(best_fit)
plot_points(features, hidden_truth)

ax.scatter(*best_fit.means_.T, color="white", marker="*", s=300)
ax.scatter(*best_fit.means_.T, color="black", marker="*", s=100, label="cluster centers")

ax.legend(loc="lower left")

plt.show()
```

This Gaussian mixture model is more effective at discovering the true penguin species because the species' distributions are not round in bill length and bill depth, but somewhat more ellipsoidal.

Now imagine that penguins in the antarctic don't come labeled with species names (which is true). What compels us to invent these labels and put real things (the live penguins) into these imaginary categories (the species)? I would say it's because we see patterns of density and sparsity in their distributions of attributes. Clustering formalizes this idea, including the ambiguity in the number of clusters and the points at the edges between clusters. All categories in nature larger than the fundamental particles are formed by some kind of clustering, formal or informal.

+++

## Hierarchical clustering

+++

Instead of specifying the number of clusters, we could have specified a cut-off threshold: penguins are considered distinct if their distance in bill length, bill depth space is larger than some number of millimeters. This is called hierarchical or agglomerative clustering.

```{code-cell} ipython3
from sklearn.cluster import AgglomerativeClustering
```

```{code-cell} ipython3
best_fit = AgglomerativeClustering(n_clusters=None, distance_threshold=30).fit(features)
```

This algorithm doesn't have a `predict` method, since it's a recursive algorithm in which adding a point to a cluster would change how additional points are added to the cluster, but we can get the cluster assignments for the input data.

```{code-cell} ipython3
best_fit.labels_
```

A distance threshold of $\mbox{30 mm}$ results in

```{code-cell} ipython3
len(np.unique(best_fit.labels_))
```

unique clusters. Here's a plot:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 5))

for label in np.unique(best_fit.labels_):
    ax.scatter(*features[best_fit.labels_ == label].T)

ax.set_xlim(29, 61)
ax.set_ylim(12, 22)
ax.set_xlabel("bill length (mm)")
ax.set_ylabel("bill depth (mm)")

plt.show()
```

This algorithm has more parameters than k-means and Gaussian mixtures. You have to provide:

* a metric for the distance between two points, $\vec{x}_{i}$ and $\vec{x}_{i'}$ ($i' \ne i$),
* a "linkage," which is the distance between a point $\vec{x}_i$ and a cluster $C_j$, which can be
  - single: the minimum distance between $\vec{x}_i$ and any $\vec{x}_{i'} \in C_j$, which tends to make long, connected worms
  - complete: the maximum distance between $\vec{x}_i$ and all $\vec{x}_{i'} \in C_j$, which tends to make round balls
  - average: the average distance between $\vec{x}_i$ and all $\vec{x}_{i'} \in C_j$
  - Wald: a measure that minimizes the variance within a cluster (used by default by Scikit-Learn, above).

I'm mentioning hierarchical clustering because it is important for HEP: jet-finding is an implementation of hierarchical clustering with HEP-specific choices for the measures above. In the FastJet manual ([ref](https://fastjet.fr/)), you'll find that the distance between two particles $i$ and $i'$ in the anti-kT algorithm is

$$d_{ii'} = \mbox{min}\left(\left(\frac{1}{p_{Ti}}\right)^2, \left(\frac{1}{p_{Ti'}}\right)^2\right) \frac{(\eta_i - \eta_{i'})^2 + (\phi_i - \phi_{i'})^2}{(\Delta R)^2}$$

where $p_{Ti}$, $\eta_i$, and $\phi_i$ are the transverse momentum, pseudorapidity, and azimuthal angle of particle $i$, respectively, and similarly for $i'$. The $\Delta R$ parameter is a user-chosen jet scale cut-off. The linkage is the distance between an unclustered particle $i$ and the vector-sum of particles in the cluster as "pseudojet" $i'$. One more complication: there's also a special "beam jet" whose distance from particle $i$ is

$$d_{iB} = \left(\frac{1}{p_{Ti}}\right)^2$$

and it is usually ignored (so all good jets are far from the QCD background expected along the beamline). Apart from these choices, HEP jet-finding is standard hierarchical clustering.

+++

## Neural networks

+++

So far, all of the clustering algorithms that I've shown are not neural networks. Deep learning approaches to the same problem are covered in the [next section](23-autoencoders.md).
