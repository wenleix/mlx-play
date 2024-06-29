import time

import mlx.core as mx
import matplotlib.pyplot as plt
import mlx.nn as nn
import math
import mlx.optimizers as optim



NUM_FEATURES = 100
NUM_EXAMPLES = 1_100
#NUM_ITERS = 10_000
NUM_ITERS = 800

lr = 0.01   # hmm~

class SimpleLinear(nn.Module):
    # y = x @ W.T + b
    # where x is input_dim, y is output_dim, ans W has shape [output_dim, input_dim]
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()

        # i guess it's important for convergence in gradient descent
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(output_dims, input_dims)
        )


    def  __call__(self, x: mx.array) -> mx.array:
        # hmm, interesting convention on "dict-like" experience
        return x @ self["weight"].T


# Groundtruth
mx.random.seed(42 ** 3)
w_star = mx.random.normal(shape=(NUM_FEATURES, 1))

# Input features and generate y by adding a bit noise :)
X = mx.random.normal(shape=(NUM_EXAMPLES, NUM_FEATURES))
y = X @ w_star + mx.random.normal(shape=(NUM_EXAMPLES, 1)) / 100.0

model = SimpleLinear(NUM_FEATURES, 1)
optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)


def loss_fn(model, X, y):
    return nn.losses.mse_loss(model(X), y, reduction="mean")

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

losses_per_step = []
for _ in range(100):
    loss, grad = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state)

    losses_per_step.append(loss.item())

plt.plot(losses_per_step,
         # marker='o'
         )
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
