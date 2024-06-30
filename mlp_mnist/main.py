import numpy as np
import mlx.core as mx
import mnist
import mlx.nn as nn
import math
import mlx.optimizers as optim
import time


SEED = 42 ** 3
NUM_LAYERS = 2
HIDDEN_DIM = 32
NUM_CLASSES = 10
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.1

nn.Linear
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
        self.bias = mx.random.uniform(
            low=-scale, high=scale, shape=(output_dims, )
        )

    def __call__(self, x: mx.array) -> mx.array:
        # hmm, interesting convention on "dict-like" experience
        return x @ self["weight"].T + self["bias"]


class SimpleMLP(nn.Module):
    # Simple MLP, use nn.Module abstraction
    # Somewhat feels "operator/kernel" in ML framework correspoding to "UDF" in database systems,
    #   and nn.Module corresponding to "Operators" in database/dataflow systems

    def __init__(
            self, num_layers, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # is nn.Module.layers special in nn.Module??
        self.layers = []
        # input -> first hidden layer
        self.layers.append(SimpleLinear(input_dim, hidden_dim))
        # (num_layers - 1) hidden layers
        for layer_id in range(num_layers - 1):
            self.layers.append(SimpleLinear(hidden_dim, hidden_dim))
        # last hideen layer -> output
        self.layers.append(SimpleLinear(hidden_dim, output_dim))

    def __call__(self, x: mx.array):
        for (index, l) in enumerate(self.layers):
            if index != 0:
                x = nn.relu(x)
            x = l(x)
        return x


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction='mean')


def batch_iterate(batch_size, images, labels):
    assert images.shape[0] == labels.shape[0]
    n = labels.shape[0]
    perm = mx.array(np.random.permutation(n))
    for i in range(0, n, batch_size):
        ids = perm[i:i + batch_size]
        yield images[ids], labels[ids]


def main():
    np.random.seed(SEED)

    # load the data
    train_images, train_labels, test_images, test_labels = map(
        mx.array, mnist.mnist()
    )
    print(train_images.shape, train_labels.shape)

    # initialize the model
    input_dim = train_images.shape[1]
    model = SimpleMLP(NUM_LAYERS, input_dim, HIDDEN_DIM, NUM_CLASSES)
    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate=LEARNING_RATE)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for e in range(NUM_EPOCHS):
        tic = time.perf_counter()
        loss = None
        for X, y in batch_iterate(BATCH_SIZE, train_images, train_labels):
            # each micro batch (X, y)
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.state)

        # evalute the model a after each epoch
        # do inference on test data
        test_preds = mx.argmax(model(test_images), axis=1)
        accuracy = mx.mean(test_preds == test_labels)
        toc = time.perf_counter()

        print(f"Epoch {e + 1}/{NUM_EPOCHS}, Loss: {loss.item()}, Accuracy: {accuracy.item() * 100:.2f}%, Time: {toc - tic:.2f}s")



if __name__ == '__main__':
    # mx.set_default_device(mx.cpu)
    main()
