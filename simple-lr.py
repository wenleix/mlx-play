# Based on https://ml-explore.github.io/mlx/build/html/usage/compile.html#compiling-training-graphs
# Binary cross entroy -> logistic regression

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# 4 examples with 10 features each
x = mx.random.uniform(shape=(4, 10))

# groundtruth w, b
w = mx.random.uniform(-1, 1, shape=(10, ))
b = mx.random.uniform(-1, 1, shape=(1, ))

# groundtruth y
y_con = x @ w.T + b
# y_dis is 0 or 1
y_dis = (y_con > 0.5).astype(mx.float32)



# simple linear model
model = nn.Linear(10, 1)

# SGD with momentum
optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)

def loss_fn(model, x, y):
    logits = model(x).squeeze()
    return nn.losses.binary_cross_entropy(logits, y)


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# 10 steps for SGD
for _ in range(10):
    loss, grads = loss_and_grad_fn(model, x, y_dis)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
