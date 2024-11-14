#!/usr/bin/env python3

import pickle
import os

index_number = 10
MAX_BLOCKS = 5

groups = index_number // (MAX_BLOCKS - 1)
start = 0
count = 1
# Loops while until the are no more groups and everything is together
while groups > 0:
    last_free = 0
    start = 0
    print("groups: ", groups)
    # We iterate over all the groups in the current L
    for i in range(groups + 1):
        print("ig: ", i)
        print("start: ", start)

        # Divides the contents in half so that we read both groups which are
        # being merged at the same time. The selection doesn't work for now.
        for j in range(
            start, min(start + (MAX_BLOCKS - 1) * count // 2, index_number // 2)
        ):
            print("j1: ", j)
            print("j2: ", j)
            # Here we should be adding to the heap from the docs until the
            # memory is full. Then we start writing. Needs more thinking
        start += (MAX_BLOCKS - 1) * count
    groups //= 2
    count += 1

with open(os.path.join("ftest", "index_0.dat"), "rb") as f:
    combined = pickle.load(f)

# Merge with files
# Code removed for being ugly and not working
