# Deep learning for particle physicists

This book is an introduction to modern neural networks (deep learning), intended for particle physicists. Deep learning is a popular topic, so there are many courses on it, each assuming a different level of mathematical background. These days, most particle physicists need to use machine learning and particle physicists have a unique combination of mathematical knowledge and practical familiarity with statistics, but generally not computer science. It can be hard to find machine learning courses that both use what a physicist knows to shorten some explanations and go deeper in others, while also not assuming what a typical physicist does not know.

This book is "introductory" because it emphasizes the fundamentals of what neural networks are, why they work, and practical steps to train neural networks of any topology. It does not get into the (changing) world of network topologies or designing new kinds of neural networks to fit new problems.

This book is an expansion of a course first presented at [CoDaS-HEP](https://codas-hep.org/) in 2024: [jpivarski-talks/2024-07-24-codas-hep-ml](https://github.com/jpivarski-talks/2024-07-24-codas-hep-ml). It also has roots in [jpivarski-talks/2024-07-08-scipy-teen-track](https://github.com/jpivarski-talks/2024-07-08-scipy-teen-track) (though that was for an entirely different audience). I am writing it in book format, rather than simply depositing my slide PDFs and Jupyter notebooks in [https://hsf-training.org/](https://hsf-training.org/), because the original format assumes that I'll verbally fill in the gaps. This format is good for two purposes:

* offline self-study for a student without a teacher, and
* for teachers preparing new course PDFs and notebooks (without having to read my mind).

These course materials include some inline problems, intended for active learning during a lecture, and a large project designed for students to work on for two hours. (In practice, experienced students finished it in an hour and beginners could have used a little more time.)

Also, this course uses two machine learning libraries for examples and problem sets: [Scikit-Learn](https://scikit-learn.org/) and [PyTorch](https://pytorch.org/). Along with [TensorFlow](https://www.tensorflow.org/), these are the machine learning libraries that most particle physicists use in 2023 (analyzed using the methodology described in [this GitHub repo](https://github.com/jpivarski-talks/2023-05-09-chep23-analysis-of-physicists) and [this talk](https://indico.jlab.org/event/459/contributions/11547/)):

![](img/github-ml-package-cmsswseed.svg){. width=800px}

The landscape of machine learning libraries is more stable than it was in the early years (2015‒2018), so this is likely a future-proof choice.

```{tableofcontents}
```
