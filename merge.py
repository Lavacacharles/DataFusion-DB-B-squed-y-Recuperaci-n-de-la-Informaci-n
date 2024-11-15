#!/usr/bin/env python3

import pickle
import os
from collections import defaultdict
import heapq

index_number = 8
MAX_BLOCKS = 3
MAX_DOCS = 500

groups = index_number // (MAX_BLOCKS - 1)
start = 0
count = 0
# Loops while until the are no more groups and everything is together
while groups > 0:
    last_free = 0
    start = 0
    print("groups: ", groups)
    # We iterate over all the groups in the current L
    for i in range(groups):
        print("ig: ", i)
        print("start: ", start)

        # Divides the contents in half so that we read both groups which are
        # being merged at the same time. The selection doesn't work for now.

        print("limit: ", start + (MAX_BLOCKS - 1) * 2**count)
        j2 = start + (MAX_BLOCKS - 1) * 2**count // 2
        for j in range(
            start, min(start + (MAX_BLOCKS - 1) * 2**count // 2, index_number)
        ):
            print("j: ", j, " - j2: ", j2)
            j2 += 1
            # Here we should be adding to the heap from the docs until the
            # memory is full. Then we start writing. Needs more thinking
        start += (MAX_BLOCKS - 1) * 2**count

    print("Extra")
    for j in range(start, index_number):
        print("j: ", j)
        # Here we should be adding to the heap from the docs until the
        # memory is full. Then we start writing. Needs more thinking
    groups //= 2
    count += 1


print("Test with lists")
# Merge with list for testing
from data import data, out

groups = index_number // (MAX_BLOCKS - 1)
start = 0
count = 0

# Loops while until the are no more groups and everything is together
while groups > 0:
    last_free = 0
    start = 0
    print("groups: ", groups)
    # We iterate over all the groups in the current L
    for i in range(groups):
        print("ig: ", i)
        print("start: ", start)

        # Divides the contents in half so that we read both groups which are
        # being merged at the same time. The selection doesn't work for now.

        print("limit: ", start + (MAX_BLOCKS - 1) * 2**count)
        heap = []
        blocks_opened = 0
        j2 = start + (MAX_BLOCKS - 1) * 2**count // 2
        for j in range(
            start, min(start + (MAX_BLOCKS - 1) * 2**count // 2, index_number)
        ):
            blocks_opened += 2
            print("j: ", j, " - j2: ", j2)
            for d in (data[j], data[j2]):
                print(d)
                for key, value in d.items():
                    heapq.heappush(heap, (key, value))
            if blocks_opened + 1 > MAX_BLOCKS:
                while True:
                    if last_free >= blocks_opened:
                        break
                    else:
                        index = defaultdict(list)
                        while True:
                            # activates if it's full, to save index
                            if True:
                                data[last_free] = index
                                break
                            else:
                                while True:
                                    top = heapq.heappop(heap)
                                    term = top[0]
                                    docs = top[1]
                                    index[term].extend(docs)
                                    if term != min(heap)[0]:
                                        index[term].sort()
                                        break
                        last_free += 1
                        blocks_opened -= 1
                for i in range(blocks_opened // 2):
                    while True:
                        top = heapq.heappop(heap)
                        term = top[0]
                        docs = top[1]
                        index[term].extend(docs)
                        if term != min(heap)[0]:
                            # We are in the next word, so sort for safety
                            index[term].sort()
                        # If the index is at the limit, save it
                        if True:
                            data[last_free] = index
                            last_free += 1
                # save to index

            j2 += 1
            # Here we should be adding to the heap from the docs until the
            # memory is full. Then we start writing. Needs more thinking
        start += (MAX_BLOCKS - 1) * 2**count

    print("Extra")
    for j in range(start, index_number):
        print("j: ", j)
        # Here we should be adding to the heap from the docs until the
        # memory is full. Then we start writing. Needs more thinking
    groups //= 2
    count += 1
