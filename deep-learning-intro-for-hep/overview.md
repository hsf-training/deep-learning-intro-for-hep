---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Overview

## Broad overview

The upshot of the machine learning revolution is that we now have two ways to make algorithms:

* craftsmanship, by which I mean hand-written code, and
* farming, which is machine learning.

![](img/craftsmanship.jpg){. width=400px} ![](img/farming.jpg){. width=400px}

Machine learning will not make hand-written programs obsolete any more than farming made manual tool-building obsolete: the two methods of development have different properties and the most appropriate applications of each don't entirely overlap.

Programming by hand allows for more precise control of the finished product, but the complexity of hand-written programs are fundamentally limited by a human mind or a team's ability to communicate. Encapsulation, specifications, and protocols help ever-larger teams work together on shared programming projects, but they do so by simplifying the interfaces, and there are limits to how simple some problems can be cast.

Machine learning, on the other hand, allows for extremely nuanced solutions. Machine learning algorithms are developed by allowing enormous numbers of parameters to fluctuate, biased toward configurations that solve the problem at hand. By analogy, living systems randomly sample the space of possible configurations, biased toward configurations that survive or are cultivated by human farmers, and thus the anatomy of a plant is far more intricate than any human could invent. Although we steer this process toward preferred outcomes, we don't control it in detail.

Thus, simple or moderately complex problems that need to be controlled with precision are best solved by hand-written algorithms, while machine learning is best for extremely complex problems or problems that can't be solved (as accurately) any other way. It is entirely reasonable for hand-written and machine learning algorithms to be used in the same workflow—in fact, it's common for machine learning algorithms to be encapsulated within hand-written pipelines, which deliver machine learning predictions where they need to go, and perhaps control for unexpected outputs from the machine learning model.

## Terminology

Some authors make distinctions between the terms

* Artificial Intelligence (AI) and
* Machine Learning (ML).

I have not seen practicioners of AI/ML distinguish these terms consistently, so I treat them as synonymous. Moreover, these two terms and

* data mining

are all applied to many-parameter fits of large datasets, and "machine learning" and "data mining" were both introduced in the decades when "artificial intelligence" was out of favor as a funded research topic—as a way to continue the research under a different name. As data analysis techniques, all three may be considered synonymous, but "data mining" wouldn't be used to describe generative techniques, such as simulating chat text, images, or physics collision events, the way that "artificial intelligence" and "machine learning" would.

These terms describe the following procedure:

1. write an algorithm (a parameterized "model") that generates output that depends on a huge number of internal parameters;
2. vary those parameters ("train the model") until the algorithm returns expected results on a labeled dataset ("supervised learning") or until it finds patterns according to some desired metric ("unsupervised learning");
3. either apply the trained model to new data, to describe the new data in the same terms as the training dataset, or use the model to generate new data that is plausibly similar to the training dataset.

Apart from the word "huge," this is curve-fitting, which most experimental physicists use on a daily basis. Consider a dataset with two observables (called "features" in ML), $x$ and $y$, and suppose that they have an approximate, but not exact, linear relationship. There's an algorithm to compute the best fit of $y$ as a function of $x$, and this linear fit is a model with two parameters: the slope and intercept of the line. If $x$ and $y$ have a non-linear relationship expressed by $N$ parameters, a non-deterministic optimizer like Minuit can be used to search for the best fit.

ML fits differ from simple curve-fitting in that the number of parameters is huge and, while the parameters are often individually meaningful in curve-fitting, the parameters of an ML model are (usually) not meaningful in themselves, only as a means to get predictions or generate data from the model.






## Goals of this course



