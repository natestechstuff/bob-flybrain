# Third-party notices

## Drosophila brain model (Shiu et al.)
Bob's neuron model and its parameters (leaky integrate-and-fire, the synapse weight, delays, refractory
period and Poisson input settings in `flyengine.py`) follow the model in
https://github.com/philshiu/Drosophila_brain_model . `flyengine.py` is our own numpy rewrite of that model,
not a copy of its code. The data download in `get_data.sh` fetches `Completeness_783.csv` and
`Connectivity_783.parquet` from that repository. Its license:

```
MIT License

Copyright (c) 2023 Philip Shiu and Nico Spiller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Paper: Shiu, P.K., Sterne, G.R., Spiller, N. et al. A Drosophila computational brain model reveals sensorimotor
processing. Nature 634, 210-219 (2024). https://doi.org/10.1038/s41586-024-07763-9

## FlyWire connectome (v783)
The wiring Bob runs on comes from the FlyWire whole-brain connectome of an adult female fruit fly. The FlyWire
data is shared by its authors under the Creative Commons Attribution-NonCommercial 4.0 license (CC BY-NC 4.0),
https://creativecommons.org/licenses/by-nc/4.0/ . None of it is stored in this repository except
`bob_fly/bob_fly.npz`, which holds a small piece of it (about 6,300 neurons of the olfactory / mushroom-body
pathway and their connections, plus Bob's trained weights); that piece stays under CC BY-NC 4.0.

Please cite:
- Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. Nature 634, 124-138 (2024).
- Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature 634, 139-152 (2024).
- Matsliah, A. et al. Neuronal parts list and wiring diagram for a visual system. Nature 634, 166-180 (2024).

## FlyWire annotations
Cell classes and types (Kenyon cells, MBONs, projection neurons, ...) come from
https://github.com/flyconnectome/flywire_annotations (Supplemental_file1_neuron_annotations.tsv), downloaded by
`get_data.sh`. See that repository for its terms and citations.

## Python packages
numpy (BSD-3-Clause), pandas (BSD-3-Clause), pyarrow (Apache-2.0).
