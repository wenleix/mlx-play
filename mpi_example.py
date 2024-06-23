import mlx.core as mx

world = mx.distributed.init()
print(world.rank(), world.size())

#x = mx.distributed.all_sum(mx.ones(10))
#print(world.rank(), x)
