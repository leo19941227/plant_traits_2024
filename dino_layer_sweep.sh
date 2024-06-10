#!/bin/bash

set -e
set -x

for layer in 0 3 6 9 12 15 18 21 24;
do
    python3 dino_hf_mlp.py submissions/dino_layer_sweep/layer_${layer} --use_hidden_state ${layer}
done

