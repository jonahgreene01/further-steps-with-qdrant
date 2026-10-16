## Task 1 Solution
### Observed values

```
Average precision@10: 1.0000
Average ANN query time: 55.22 ms
Average exact k-NN query time: 301.29 ms
```

### Reflection

The approximate search got a precision@10 of 1.0, so for all 100 test queries it returned the same 10 papers as the exact search. Since the exact top 10 is the golden set and both lists have 10 items, recall@10 is also 1.0 here. I expected high precision, but not a perfect score, since ANN is supposed to be close to the true neighbors, not guaranteed to match them.

To see if it ever misses, I reran it with k = 20 and got 0.9995. That works out to 1,999 of 2,000 true neighbors found: one query got 19 of its 20, and the other 99 got all 20. So misses do happen, but they're rare with Qdrant's default settings, and I only saw one once I asked for more results per query.

For speed, ANN averaged 55.22 ms and exact search averaged 301.29 ms, so exact took about 5.5 times as long. That makes sense: exact search compares the query against all 411,120 vectors, while HNSW follows links through its graph and only checks part of the collection. Since exact search has to look at every vector, I'd expect the gap to get bigger as the collection grows.

A few things to keep in mind:
- The times include the whole round trip from my script to Qdrant and back, not just the search itself.
- This was one run on my machine.
- 100 queries is a small test set. Getting 1.0 means ANN didn't miss on these queries, not that it never misses.

Overall, for this collection the default approximate search gave the same results as exact search for these queries in a fraction of the time, so it looks like the right choice.