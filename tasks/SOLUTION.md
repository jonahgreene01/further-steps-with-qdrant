## Task 2

### Observed values

```
hnsw_ef= 10: avg precision@10 = 0.9670, avg query time = 13.66 ms
hnsw_ef= 20: avg precision@10 = 0.9900, avg query time = 12.82 ms
hnsw_ef= 50: avg precision@10 = 1.0000, avg query time = 15.60 ms
hnsw_ef=100: avg precision@10 = 1.0000, avg query time = 16.20 ms
hnsw_ef=200: avg precision@10 = 1.0000, avg query time = 17.86 ms
```

### Reflection

Precision goes up with hnsw_ef and then levels off. At ef=10 it was 0.967, at ef=20 it was 0.990, and from ef=50 up it was a perfect 1.0. This makes sense: hnsw_ef is the size of the candidate list HNSW keeps while it searches the graph. With only 10 or 20 candidates, the search sometimes stops before it reaches all of the true nearest neighbors. With 50 or more, it explored enough of the graph to find all of them for every query in this set.

Query time mostly goes up with hnsw_ef, from 12.82 ms at ef=20 to 17.86 ms at ef=200. That's the cost of keeping a longer candidate list and checking more nodes. ef=10 was slightly slower than ef=20, but the difference is under 1 ms, so I don't think it means much. The differences overall are only a few milliseconds, which is small compared to the total time of each request. Timing each value only once gave noisy results with no clear pattern, so these times are averaged over repeated runs.
The main takeaway is that ef=50 was the lowest value that found every true neighbor on this test set. Going higher than that just added time without improving precision.

