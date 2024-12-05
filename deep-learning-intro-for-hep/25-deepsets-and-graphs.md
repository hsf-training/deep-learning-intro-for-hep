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

# Ragged data and Graph Neural Networks (GNNs)

+++



```{code-cell} ipython3
import numpy as np
import awkward as ak
import uproot

import torch
from torch import nn, optim
from torch_geometric.nn import aggr
```

```{code-cell} ipython3
event_data = uproot.open("data/SMHiggsToZZTo4L.root")["Events"].arrays()
```

```{code-cell} ipython3
event_data.Muon_pt
```

```{code-cell} ipython3
event_data.fields
```
