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

# History of HEP and ML

The purpose of this section is to show that particle/High Energy Physics (HEP) has always needed ML. It is not a fad—ML would have brought major benefits to HEP at any stage in its history. It's being used now because it's just becoming _possible_ now.

![](img/hep-plus-ml.jpg){. width="40%"}

## HEP, computation, and big data

Big data processing has always been an essential part of HEP, since the beginning of the field. The late 1940's/1950's can be called the "beginning" of HEP because that's when it acquired the characteristics that define it today.

1. Physicists in this era used accelerators to collide subatomic particles with more energy and higher flux than is observed in nature. Then, as now, cosmic rays produce higher-energy collisions than can be produced by accelerators, but not with high enough rates to study those events in detail. Ernest Lawrence's large team at Berkeley (dozens of scientists) invented a series of ever-larger accelerators in the 1930's, but it was in the late 1940's that accelerators started to be built around the world as the preferred tool of discovery.
2. These collisions produced new particles to discover, for any type of particle whose mass is less than the center-of-mass energy of the collision (also constrained by quantum numbers). In the first few decades, the accelerated particles were driven into stationary targets, rather than two colliding beams with a higher center-of-mass. However, the goal of the experiments was to produce particles that can't ordinarily be found in nature and study their properties, just as it is now.
3. Computers quantified particle trajectories, reconstructed invisible (neutral) particles, and rejected backgrounds, for as many collision events as possible.

The last point is key: high energy physicists started using computers as soon as they became available. For example, Luis Alvarez's group at Berkeley's \$9,000,000 Bevatron built a \$1,250,000 detector experiment and bought a \$200,000 IBM 650 to analyze the data in 1955 ([ref](https://www2.lbl.gov/Science-Articles/Research-Review/Magazine/1981/81fchp6.html)). Computers were a significant fraction of the already large costs, and yet analysis productivity was still limited by processing speed.

![](img/overall-view-of-bevatron-magnet-photograph-taken-september-6-1955-bevatron-088cb0-1600.jpg){. width="31%"} ![](img/alvarez-group-bubble-chamber.jpg){. width="28%"} ![](img/ibm-650.jpg){. width="39%"}

The limitation was algorithmic: physicists wanted to compute kinematics of the observed particles, which is easy (just a formula), but the particles were observed as images, which are hard to convert into trajectories. Something has to find the locations of the line segments and endpoints in the photograph. For the first 10‒20 years, that "something" was people: humans, mostly women, identified the vertices and trajectories of tracks in bubble chamber photos on specially designed human-computer interfaces (see [ref](https://www.physics.ucla.edu/marty/HighEnergyPhysics.pdf) for a first-hand account).

![](img/franckenstein-3.jpg){. width="50%"}

This is a pattern-recognition task—if ML had been available (in a usable form) in the 1950's, then it would have better than using humans for this task. Moreover, humans couldn't keep up with the rate. Then as now, the quality of the results—discovery potential and statistical precision—scales with the number of analyzed events: the more, the better. The following plot (from <a href="https://books.google.de/books?id=imidr-iFYCwC&lpg=PA129&dq=jack%20franck%20franckenstein&pg=PA130#v=onepage&q&f=false">ref</a>) was made to quantify the event interpretation rate using different human-computer interfaces.

![](img/scaleup.png){. width="60%"}

Below, I extended the plot to the present day: the number of events per second has continued to increase exponentially.

![](img/event-rates.svg){. width="100%"}

These event rates have been too fast for humans since the 1970's, when human scanners were replaced by heuristic track-finding routines, usually by hand-written algorithms that iterate through all combinations within plausible windows (which are now a bottleneck in high track densities).

Although many computing tasks in particle physics are suitable for hand-written algorithms, the field also has and has always had tasks that are a natural fit for ML and artificial intelligence, to such an extent that human intelligence was enlisted to solve them. While ML would have been beneficial to HEP from the very beginning of the field, algorithms and computational resources have only recently made it possible.

## Summers and winters of AI

AI has had a rocky history, driven by competition between two philosophies that have been called "symbolic" (or "GOFAI: Good Old Fashioned AI") and "connectionist."

Symbolic AI consists of hand-written algorithms, which today wouldn't be called AI. In fact, it was a joke that once a human-like problem had been solved by computer, at that point [it ceases to be AI](https://quoteinvestigator.com/2024/06/20/not-ai/). The problems that went through this cycle include:

* parsing the source code of a high-level programming language,
* checking the proof of a theorem, and later, searching for proofs,
* playing chess and other games by truncated searches through all possible games,
* expert systems and fuzzy database searches.

Symbolic AI is called "symbolic" because the starting point is a system of abstract symbols and rules—like programming in general. An associated philosophical idea is that this is the starting point for intelligence itself: human and artificial intelligence consists in manipulating propositions like an algebra, as proposed by George Boole, Gottlob Frege, and Bertrand Russell, and in language by Noam Chomsky.

Connectionist AI makes a weaker claim about what happens in the intelligent system (human or artificial): only that the system's inputs and outputs are correlated appropriately, and an intricate network of connections can implement this correlation. As we'll see, neural networks are an effective way to implement it, and they were (loosely) inspired by the biology of human brains. The idea that we can only talk about the inputs and outputs of human systems, without proposing entities like symbols in the mind, was a popular trend in psychology called Behaviorism in the first half of the 20th century (in a similar spirit as Quantum Mechanics and S-Matrix theory in physics, in which we only speak of particles going in and particles coming out of an interaction, not what happens in between). Today, Cognitive Psychologists can measure properties of algorithms in human minds, so Behaviorism is out of favor. It's ironic that large language models like ChatGPT are an implementation of what Behaviorists proposed as a model of human intelligence, decades ago.
