

import os
import numpy as np

from einops import rearrange, reduce, repeat
# from einops.layers.torch import Rearrange

# ----------
# REFERENCE
# https://cpp-learning.com/einops/


# -------------------------------------------------------------------------------------------------
# rearrange
# -------------------------------------------------------------------------------------------------

x = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]],
    [[13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24]]])


print(x.shape)


# ----------
y = rearrange(x, 'a b c -> a c b')

print(y)
print(x.shape)
print(y.shape)


# ----------
y = rearrange(x, 'a b c -> a (b c)')
print(y)
print(x.shape)
print(y.shape)


y = rearrange(x, 'a b c -> (a b c)')
print(y)
print(x.shape)
print(y.shape)


# -------------------------------------------------------------------------------------------------
# reduce
# -------------------------------------------------------------------------------------------------

y1 = reduce(x, 'a b c -> b c', 'max')
y2 = reduce(x, 'a b c -> b c', 'min')

print(y1) # (a, b, c)から(b, c)に再配置して、大きい数値のみ採用
print(y2) # (a, b, c)から(b, c)に再配置して、小さい数値のみ採用


# ----------
# Sequential
input = x
x1 = rearrange(input, 'a b c -> a c b')
x2 = reduce(x1, 'a c b -> c b', 'min')
output = reduce(x2, 'c b -> ', 'sum')
print("input -> x1 -> x2 -> output =", output)


# -------------------------------------------------------------------------------------------------
# repeat
# -------------------------------------------------------------------------------------------------

y3 = repeat(x2, 'c b -> a c b', a=3)
print(y3) # (c, b)をaの数だけコピーして、増やした軸に再配置


