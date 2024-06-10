#!/bin/bash

set -x
set -e

for iteration in 10 50 100 150 200;
do
    python3 dino_boosting.py submissions/filter_5_95/iteration_${iteration} --n_iter ${iteration} --filter_outlier --filter_low 0.051 --filter_high 0.951
done

