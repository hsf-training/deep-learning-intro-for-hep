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

![](img/craftsmanship.jpg){. width="49%"} ![](img/farming.jpg){. width="49%"}

Machine learning will not make hand-written programs obsolete any more than farming made manual tool-building obsolete: the two methods of development have different strengths and the most appropriate applications of each don't entirely overlap.

Programming by hand allows for more precise control of the finished product, but the complexity of hand-written programs is fundamentally limited by a human mind or a team's ability to communicate. Encapsulation, specifications, and protocols help ever-larger teams work together on shared programming projects, but they do so by simplifying the interfaces, and there are limits to how simple some problems can be cast.

Machine learning, on the other hand, allows for extremely nuanced solutions. Machine learning algorithms are developed by allowing enormous numbers of parameters to fluctuate, biased toward configurations that solve the problem at hand. By analogy, living systems randomly sample the space of possible configurations, biased toward configurations that survive (natural evolution) or are selected by human farmers (cultivation), and thus the anatomy of a plant is more intricate than any human could ever invent. Although we steer this process toward preferred bulk properties, we don't control it in detail.

Thus, simple or moderately complex problems that need to be controlled with precision are best solved by hand-written programs, while machine learning is best for extremely complex problems or problems that can't be solved (or solved as accurately) any other way. It is entirely reasonable for hand-written and machine learning algorithms to coexist in the same workflow—in fact, it's common for machine learning algorithms to be encapsulated in conventional frameworks, which deliver the machine learning outputs where they need to go, and perhaps adjust for unexpected outputs when the machine learning algorithm goes awry.

## Terminology

Some authors make distinctions between the terms

* Artificial Intelligence (AI) and
* Machine Learning (ML).

I have not seen practicioners of AI/ML distinguish these terms consistently, so I treat them as synonymous. Moreover, these terms and

* data mining
* MultiVariate Analysis (MVA)

are all applied to many-parameter fits of large datasets. "Machine learning," "data mining," and "multivariate analysis" were all introduced during the decades when "artificial intelligence" was out of favor as a funded research topic—as a way to continue the research under different names. As data analysis techniques, all of these words may be considered synonymous, but "data mining" and "multivariate analysis" wouldn't be used to describe _generative_ techniques, such as simulating chat text, images, or physics collision events, the way that "artificial intelligence" and "machine learning" are.

All of these terms describe the following general procedure:

1. write an algorithm (a parameterized "model") that generates output that depends on a huge number of internal parameters;
2. vary those parameters ("train the model") until the algorithm returns expected results on a labeled dataset ("supervised learning") or until it finds patterns according to some desired metric ("unsupervised learning");
3. either apply the trained model to new data, to describe the new data in the same terms as the training dataset ("predictive"), or use the model to generate new data that is plausibly similar to the training dataset ("generative"; AI and ML only).

Apart from the word "huge," this procedure also describes curve-fitting, a ubiquitous analysis technique that most experimental physicists use on a daily basis. Consider a dataset with two observables (called "features" in ML), $x$ and $y$, and suppose that they have an approximate, but not exact, linear relationship. There is [an exact algorithm](https://en.wikipedia.org/wiki/Linear_regression#Formulation) to compute the best fit of $y$ as a function of $x$, and this linear fit is a model with two parameters: the slope and intercept of the line. If $x$ and $y$ have a non-linear relationship expressed by $N$ parameters, a non-deterministic optimizer like [MINUIT](https://en.wikipedia.org/wiki/MINUIT) can be used to search for the best fit.

ML fits differ from curve-fitting in the number of parameters used and their interpretation—or rather, their lack of interpretation. In curve fitting, the values of the parameters and their uncertainties are regarded as the final product, often quoted in a table as the result of the data analysis. In ML, the parameters are too numerous to present this way and wouldn't be useful if they were, since the calculation of predicted values from these parameters is complex. Instead, the ML model is used as a machine to predict $y$ for new $x$ values (prediction) or to randomly generate new $x$, $y$ pairs with the same distribution as the training set (generation). In fact, most ML models don't even have a unique minimum in parameter space—different combinations of parameters would result in the same predictions.

Today, the most accurate and versatile class of ML models are "deep" Neural Networks (NN), where "deep" means having a large number of internal layers. I will describe this type of model in much more detail, since this course will focus exclusively on them. However, it's worth pointing out that NNs are just one type of ML model; others include:

* [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier),
* [k-Nearest Neighbors (kNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm),
* [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis),
* [generalized additive models](https://en.wikipedia.org/wiki/Generalized_additive_model), such as [LOWESS fitting](https://en.wikipedia.org/wiki/Local_regression),
* [decision trees](https://en.wikipedia.org/wiki/Decision_tree), (boosted) [random forests](https://en.wikipedia.org/wiki/Random_forest), in particular [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost),
* [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering), [Gaussian mixture models](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model), [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering),
* [Gaussian processes](https://en.wikipedia.org/wiki/Gaussian_process) or "[Kriging](https://en.wikipedia.org/wiki/Kriging)",
* [Support Vector Machines (SVMs)](https://en.wikipedia.org/wiki/Support_vector_machine),
* [Hidden Markov Models (HMMs)](https://en.wikipedia.org/wiki/Hidden_Markov_model),

Boosted random forests were particularly popular in particle physics before the deep learning revolution (around 2015), and they're still widely used (through [XGBoost](https://xgboost.readthedocs.io/) and ROOT's [TMVA](https://root.cern/manual/tmva/)). Most of the above algorithms are still relevant in some domains, particularly if available datasets are too small to train a deep NN.

This course focuses on NNs for several reasons.

1. At heart, an NN is a simple algorithm, a generalization of a linear fit.
2. NNs are applicable to a broad range of problems, when large enough training datasets and computational resources are available to train them.
3. They're open to experimentation with different NN topologies.
4. At the time of writing, we are in the midst of an ML/AI revolution, almost entirely due to advances in deep NNs.

## Goals of this course

At the end of the course, I want students to

1. understand neural networks at a deep level, to know _why_ they work and therefore _when_ to apply them, and
2. be sufficiently familiar with the tools and techniques of model-building to be able to start writing code (in PyTorch). The two hour exercise asks students to do exactly this, for a relevant particle physics problem.

In particular, I want to dispel the notion that ML is a black box or a dark art. Like the scientific method itself, ML development requires tinkering and experimentation, but it is possible to debug and it is possible to make informed decisions about it.
