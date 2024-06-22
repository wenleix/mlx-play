import mlx.core as mx

state = []

@mx.compile
def fun(x, y):
    z = x + y
    state.append(z)
    return mx.exp(z)


fun(mx.array(1.0), mx.array(2.0))
print(type(state))
print(len(state))
print(type(state[0]))

# Crash!
print(state)

