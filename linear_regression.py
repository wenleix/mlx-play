# written only use the Tensor + autograd level abstraction
# doesn't use nn.Module/optimizer/common loss function abstraction

# based on https://github.com/ml-explore/mlx/blob/main/examples/python/linear_regression.py

import time

import mlx.core as mx
import matplotlib.pyplot as plt


NUM_FEATURES = 100
NUM_EXAMPLES = 1_100
#NUM_ITERS = 10_000
NUM_ITERS = 800

lr = 0.01   # hmm~

# Groundtruth
mx.random.seed(42 ** 3)
w_star = mx.random.normal(shape=(NUM_FEATURES, ))

# Input features and generate y by adding a bit noise :)
X = mx.random.normal(shape=(NUM_EXAMPLES, NUM_FEATURES))
y = X @ w_star + mx.random.normal(shape=(NUM_EXAMPLES, )) / 100.0

# Iniitalize w
w = mx.random.normal(shape=(NUM_FEATURES, ))


def loss_fn(w):
    # Sum over the squared difference
    # return mx.sum(mx.square(X @ w - y))

    # use mean instead of sum is important; otherwise perhaps the learning rate is not the right magnitude
    #   and cause oscillation or divergence
    return mx.mean(mx.square(X @ w - y))


# Well, the standard auto grad
grad_fn = mx.grad(loss_fn)

tic = time.time()
# standard training loop
losses_per_step = []

for _ in range(NUM_ITERS):
    grad = grad_fn(w)
    w -= lr * grad
    # i guess the only bad experience for "pure lazy" mode backend?
    mx.eval(w)
    # print("==========================")
    # print(w)
    # print(loss_fn(w).item())

    loss = loss_fn(w).item()
    mx.eval(loss)
    losses_per_step.append(loss)


toc = time.time()

loss = loss_fn(w)
throughput = NUM_ITERS / (toc - tic)

plt.plot(losses_per_step,
        #    marker='o',
)
# set y-axis to log scale
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()


print(f"Loss: {loss}, Loss: {loss.item()}, Throughput: {throughput} iters/sec")




