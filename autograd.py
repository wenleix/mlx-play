import mlx.core as mx


def loss_fn(w, x, y):
    return mx.mean(mx.square(w * x - y))


w = mx.array(1.0)
x = mx.array([0.5, -0.5])
y = mx.array([1.5, -1.5])

grad_fn = mx.grad(loss_fn)
dloss_dw = grad_fn(w, x, y)
print(dloss_dw)

