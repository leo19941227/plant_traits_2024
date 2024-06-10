#!/bin/bash

set -x
set -e

for iteration in 10 50 100 150 200;
do
    python3 dino_boosting.py submissions/no_filtering/iteration_${iteration} --n_iter ${iteration}
done

