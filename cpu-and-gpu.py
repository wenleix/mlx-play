import mlx.core as mx
import time

# From https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html#a-simple-example


def fun(a, b, d1, d2):
    x = mx.matmul(a, b, stream=d1)
#    for _ in range(500):
#        b = mx.exp(b, stream=d2)

    return x, b


def timeit(fun, a, b, d1, d2):
    # warmup
    for _ in range(10):
        mx.eval(fun(a, b, d1, d2))

    tic = time.perf_counter()
    for _ in range(100):
        mx.eval(fun(a, b, d1, d2))
    toc = time.perf_counter()
    tpi = 1000.0 * (toc - tic) / 100
    print(f"Time per iteration: {tpi:.3f} ms")

a = mx.random.uniform(shape=(4096, 1024))
b = mx.random.uniform(shape=(1024, 4))


timeit(fun, a, b, mx.cpu, mx.cpu)
timeit(fun, a, b, mx.gpu, mx.gpu)
timeit(fun, a, b, mx.gpu, mx.cpu)
timeit(fun, a, b, mx.gpu, mx.gpu)
timeit(fun, a, b, mx.cpu, mx.cpu)

