# Deep learning for particle physicists

This book is an introduction to modern neural networks (deep learning), intended for particle physicists. Deep learning is a popular topic, so there are many courses on it, each assuming a different level of mathematical background. Most particle physicists need to use machine learning for their analysis or detector studies, and they have a unique combination of mathematical knowledge and familiarity with statistics that can help in understanding the foundations. However, most physicists don't have the same background as computer scientists or data scientists, so it can be hard to find a machine learning course that both uses what a physicist knows to shorten some explanations and go deeper in others, while also not assuming what a typical physicist does not know.

This book is "introductory" because it emphasizes the foundations of what neural networks are, how they work, _why_ they work, and provides practical steps to train neural networks of any topology. It does not get into the (changing) world of network topologies or designing new kinds of machine learning algorithms to fit new problems.

The material in this book was first presented at [CoDaS-HEP](https://codas-hep.org/) in 2024: [jpivarski-talks/2024-07-24-codas-hep-ml](https://github.com/jpivarski-talks/2024-07-24-codas-hep-ml). It also has roots in [jpivarski-talks/2024-07-08-scipy-teen-track](https://github.com/jpivarski-talks/2024-07-08-scipy-teen-track) (though that was for an entirely different audience). I am writing it in book format, rather than simply depositing my slide PDFs and Jupyter notebooks in [https://hsf-training.org/](https://hsf-training.org/), because the original format assumes that I'll verbally fill in the gaps. This format is good for two purposes:

* offline self-study by a student without a teacher, and
* for teachers preparing new course slides and notebooks (without having to read my mind).

The course materials include some inline problems, intended for active learning during a lecture, and a large project designed for students to work on for about two hours. (In practice, experienced students finished it in an hour and beginners could have used a little more time.)

Also, this course uses [Scikit-Learn](https://scikit-learn.org/) and [PyTorch](https://pytorch.org/) for examples and problem sets. [TensorFlow](https://www.tensorflow.org/) is also a popular machine learning library, but its functionality mostly duplicates PyTorch, and I didn't want to hide the conceptual material behind different interfaces. (I included Scikit-Learn because it is much simpler than PyTorch, which makes it good for smaller examples.)

I believe that the choice of choice of PyTorch over TensorFlow is more future-proof. The plot below, derived using the methodology in [this GitHub repo](https://github.com/jpivarski-talks/2023-05-09-chep23-analysis-of-physicists) and [this talk](https://indico.jlab.org/event/459/contributions/11547/), shows adoption of machine learning libraries by CMS physicists, in which Scikit-Learn, PyTorch, and TensorFlow are all equally popular.

![](img/github-ml-package-cmsswseed.svg){. width="100%"}

However, beyond particle physics, <a href="https://trends.google.com/trends/explore?q=%2Fm%2F0h97pvq,%2Fg%2F11bwp1s2k3,%2Fg%2F11gd3905v1&date=2014-08-14%202024-08-14">PyTorch is more frequently the subject of Google searches than TensorFlow</a> and [PyTorch is much more frequently used in machine learning competitions than TensorFlow](https://mlcontests.com/state-of-competitive-machine-learning-2023/#deep-learning). Although [JAX](https://jax.readthedocs.io/), an array library intended to build machine learning algorithms from the ground up, is interesting, it's not yet widely used by machine learning practitioners in particle physics or beyond.

```{tableofcontents}
```
