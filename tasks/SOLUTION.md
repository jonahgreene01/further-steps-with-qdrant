## Task 3 Solution

### Observed values

```
Collection status: yellow
Collection status: yellow
Collection status: yellow
Collection status: yellow
Collection status: yellow
Collection status: yellow
Collection status: yellow
Collection status: yellow
Collection status: green
Quantization config: scalar=ScalarQuantizationConfig(type=<ScalarType.INT8: 'int8'>, quantile=0.99, always_ram=False)

rescore=True
Average precision@10: 0.9990
Average ANN query time: 45.12 ms
Average exact k-NN query time: 130.18 ms

rescore=False
Average precision@10: 0.8460
Average ANN query time: 20.53 ms
Average exact k-NN query time: 87.80 ms
```
### Reflection

With rescoring on, the quantized search was almost as accurate as the exact search, with a precision@10 of 0.999. With rescoring off, precision dropped to 0.846, so a noticeable number of the top 10 results were wrong.

This makes sense given what quantization does. Converting each number from float32 to int8 snaps it to one of 256 steps, so small differences between values get lost. When papers have very similar scores, that's enough to change their order, and some papers that should be in the top 10 get pushed out. With rescoring, the int8 vectors are only used to pick a wider pool of 20 candidates, and then the original vectors decide the final ranking. The true top 10 were almost always somewhere in that pool, so rescoring recovered nearly all of the accuracy.

The trade-off shows up in the speed. The search without rescoring averaged 20.53 ms, compared to 45.12 ms with rescoring. Skipping the step that goes back to the original vectors saves time, but costs accuracy.

The exact search times are worth noting too. The exact search is the same in both runs, since rescore only affects the quantized search, but it averaged 130.18 ms in the first run and 87.80 ms in the second. So the timings on my machine vary a lot from run to run.

Overall, rescoring looks like the right default: it keeps accuracy close to exact search. Turning it off is only worth it if speed matters more than getting the exact top results.