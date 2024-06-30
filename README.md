# Matmul

This is an experiment repo for writing fast matrix multiplication kernels on Metal.

Current speeds of a 4096x4096x4096 matmul on M1 Pro:
```
Naive: 6866 ms
Warp Coalesced: 635 ms
SMEM Tiled: 398 ms
1D Register Tiled: 240 ms
2D Register Tiled: 171 ms
SIMD: 41 ms
2D SIMD: 69 ms
SIMD Prefetch: 54 ms
MLX: 48 ms
```

The result of this exercise is a deep understanding of the kernels. This feeds into writing good compilers for [Luminal](https://github.com/jafioti/luminal).
