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

# Beyond supervised regression & classification

+++

This course is about deep learning for particle physicists, but so far, its scope has been rather limited:

1. Our neural network examples haven't been particularly deep, and some of them were linear fits (depth = 0).
2. We've dealt with only two types of problems: supervised regression and supervised classification.
3. All of the neural networks we've seen have the simplest kind of topology: fully connected, non-recurrent.

The reason for (1) is that making networks deeper is just a matter of adding layers. Deep neural networks _used to be_ hard to train, but many of the technical hurdles have been built into the tools we use: backpropagation, initial parameterization, ReLU, Adam, etc. Long computation times and (related) hyperparameter tuning are still challenging, but you can only experience them in full-scale problems. This course has focused on understanding the pieces, because that's what you'll use to find your way out of such problems.

The reason for (2) is that many (not all) HEP problems are supervised. We usually know what functions we want to approximate because we have detailed Monte Carlo (MC) simulations of the fundamental physics and our detectors, so we can use "MC truth" as targets in training. This is very different from wanting ChatGPT to "respond like a human." Sometimes, however, we want algorithms to find "unusual events" or group points into clusters, so we'll be turning to algorithms of these types soon.

For (3), there is no excuse. The fact that neural networks can be arbitrary graphs leaves them wide open to experimentation with different topologies. This is where much of the research is. In the remainder of this course, we'll look at a few topologies, but I'll mostly leave you to explore these beyond the course.
