import math
import time

import mlx.core as mx
import mlx.nn

# Example from https://ml-explore.github.io/mlx/build/html/usage/compile.html#example-speedup

# don't use mlx.nn.gelu -- it's already annotated with @partial(mx.compile, shapeless=True)
def gelu(x):
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2
    # return x * (1 + mx.erf(mx.divide(x, math.sqrt(2), stream=mx.gpu))) / 2


def timeit(fun, x):
    # warmup
    for _ in range(10):
        mx.eval(fun(x))

    tic = time.perf_counter()
    for _ in range(100):
        mx.eval(fun(x))
    toc = time.perf_counter()
    tpi = 1000.0 * (toc - tic) / 100
    print(f"Time per iteration: {tpi:.3f} ms")


x = mx.random.uniform(shape=(32, 1000, 4096))
timeit(gelu, x)
timeit(mx.compile(gelu), x)
