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
2. These collisions produced new particles to discover, for any particle whose mass is less than the center-of-mass energy of the collision (also constrained by quantum numbers). Although in the first few decades, accelerated particles were collided with stationary targets, the goal of the experiments was to produce particles that can't ordinarily be found in nature and study their properties, then as now.
3. Computers quantified particle trajectories, reconstructed invisible (neutral) particles, and rejected backgrounds, for as many collision events as possible.

The last point is key: high energy physicists started using computers as soon as they became available. For example, Luis Alvarez's group at Berkeley's \$9,000,000 Bevatron built a \$1,250,000 detector experiment and bought a \$200,000 IBM 650 to analyze the data ([ref](https://www2.lbl.gov/Science-Articles/Research-Review/Magazine/1981/81fchp6.html)). Computers were a significant fraction of the already large costs, and yet analysis productivity was still limited by processing speed.

![](img/overall-view-of-bevatron-magnet-photograph-taken-september-6-1955-bevatron-088cb0-1600.jpg){. width="31%"} ![](img/alvarez-group-bubble-chamber.jpg){. width="28%"} ![](img/ibm-650.jpg){. width="39%"}


