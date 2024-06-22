# Based on https://ml-explore.github.io/mlx/build/html/usage/compile.html#compiling-training-graphs
# Binary cross entroy -> logistic regression

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib.pyplot as plt
from functools import partial

mx.random.seed(42 ** 3)

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


# Need to specify the state when compiling the optimization step
# hmm, not the ideal dev UX...
state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

losses_per_step = []

# 10 steps for SGD
for _ in range(50):
    loss = step(x, y_dis)
    # evaluate the model and optimizer state
    mx.eval(state)
    losses_per_step.append(loss.item())


plt.plot(losses_per_step, marker='o')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

print(model.parameters())
print(model(x).squeeze())
print(y_dis)
