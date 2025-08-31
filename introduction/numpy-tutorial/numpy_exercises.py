#!/usr/bin/env python3

# NumPy quick tour with runnable, minimal examples.

import numpy as np
from numpy import random


def section(title: str):
    print(f"\n=== {title} ===")


def basics():
    section("Basics and dtypes")
    print(np.__version__)
    a = np.array([1, 2, 3, 4, 5])
    print(a, type(a))
    b = np.array([[[1, 2, 3], [4, 5, 6]],
                  [[1, 2, 3], [4, 5, 6]]])
    print("ndim:", np.array(42).ndim, np.array([1, 2]).ndim,
          np.array([[1, 2]]).ndim, b.ndim)

    c = np.array([1, 2, 3, 4], ndmin=5)
    print("ndmin=5 shape:", c.shape)

    # Indexing
    arr2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    print("2nd elem on 1st row:", arr2[0, 1])

    arr3 = np.array([[[1, 2, 3], [4, 5, 6]],
                      [[7, 8, 9], [10, 11, 12]]])
    print("3D pick [0,1,2]:", arr3[0, 1, 2])
    print("Neg index last of 2nd row:", arr2[1, -1])

    # Slicing
    s = np.array([1,2,3,4,5,6,7])
    print("slice 1:5:", s[1:5])
    print("slice -3:-1:", s[-3:-1])
    print("slice step 2 (1:5):", s[1:5:2])
    print("slice full step 2:", s[::2])
    print("row 1, 1:4:", arr2[1, 1:4])
    print("col index 2 both rows:", arr2[:, 2])

    # Dtype
    d = np.array([1,2,3,4])
    print("dtype int:", d.dtype)
    e = np.array(['apple', 'banana', 'cherry'])
    print("dtype str:", e.dtype)
    f = np.array([1,2,3,4], dtype='S')
    print("as bytes:", f, f.dtype)
    g = np.array([1,2,3,4], dtype='i4')
    print("i4:", g, g.dtype)

    # astype
    floats = np.array([1.1, 2.1, 3.1])
    print("astype('i'):", floats.astype('i'))
    print("astype(int):", floats.astype(int))
    print("astype(bool):", np.array([1,0,3]).astype(bool))


def copy_view_shape():
    section("Copy vs view, shape, reshape")
    arr = np.array([1,2,3,4,5])
    x = arr.copy()
    arr[0] = 42
    print("copy unaffected:", x)

    arr = np.array([1,2,3,4,5])
    y = arr.view()
    y[0] = 31
    print("view reflected in original:", arr)
    print("copy.base:", x.base, "view.base is original:", y.base is not None)

    m = np.array([[1,2,3,4], [5,6,7,8]])
    print("shape:", m.shape)

    v5 = np.array([1,2,3,4], ndmin=5)
    print("ndmin=5:", v5.shape)

    r = np.arange(1, 13)
    print("reshape 4x3:\n", r.reshape(4, 3))
    print("reshape 2x3x2:\n", r.reshape(2, 3, 2))
    r8 = np.arange(1, 9)
    print("reshape 2x2x-1:\n", r8.reshape(2, 2, -1))
    print("flatten with -1:", np.array([[1,2,3],[4,5,6]]).reshape(-1))


def iterate_and_nditer():
    section("Iteration")
    for x in np.array([1,2,3]): print(x, end=" ")
    print()
    mat = np.array([[1,2,3],[4,5,6]])
    for row in mat: print(row)
    for x in np.nditer(np.array([[[1,2],[3,4]], [[5,6],[7,8]]])):
        print(x, end=" ")
    print()

    # ndenumerate with indices
    for idx, val in np.ndenumerate(mat):
        print(f"{idx}:{val}", end=" ")
    print()

    # Str buffer iteration
    for x in np.nditer(np.array([1,2,3]), flags=['buffered'], op_dtypes=['S']):
        print(x, end=" ")
    print()


def stack_split():
    section("Concatenate, stack, split")
    a1, a2 = np.array([1,2,3]), np.array([4,5,6])
    print("concatenate:", np.concatenate((a1, a2)))
    print("hstack:", np.hstack((a1, a2)))
    print("vstack:\n", np.vstack((a1, a2)))
    print("dstack:\n", np.dstack((a1, a2)))

    s = np.array([1,2,3,4,5,6])
    parts = np.array_split(s, 3)
    print("array_split 3:", [p.tolist() for p in parts])

    M = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]])
    print("split rows into 3:", [p.shape for p in np.array_split(M, 3)])
    print("hsplit 3 cols:", [p.shape for p in np.hsplit(M, 3)])


def search_sort_filter():
    section("Search, sort, filter")
    arr = np.array([1,2,3,4,5,4,4])
    print("where == 4:", np.where(arr == 4))
    even_idx = np.where(np.arange(1,9) % 2 == 0)
    print("even idx:", even_idx)
    print("searchsorted 7:", np.searchsorted(np.array([6,7,8,9]), 7))
    print("searchsorted right:", np.searchsorted(np.array([6,7,8,9]), 7, side='right'))
    print("searchsorted vector:", np.searchsorted(np.array([1,3,5,7]), [2,4,6]))
    print("sort:", np.sort(np.array([3,2,0,1])))
    print("sort strings:", np.sort(np.array(['banana','cherry','apple'])))
    print("sort 2D:\n", np.sort(np.array([[3,2,4],[5,0,1]])))

    # boolean filtering
    arr = np.array([41,42,43,44])
    mask = arr > 42
    print("filter >42:", arr[mask])
    arr = np.array([1,2,3,4,5,6,7])
    print("even filter:", arr[arr % 2 == 0])


def random_sampling():
    section("Random sampling")
    print("randint 0..99:", random.randint(100))
    print("rand float [0,1):", random.rand())
    print("randint size=5:", random.randint(100, size=5))
    print("randint 3x5:\n", random.randint(100, size=(3,5)))
    print("rand(3,5):\n", random.rand(3,5))
    print("choice scalar:", random.choice([3,5,7,9]))
    print("choice 3x5:\n", random.choice([3,5,7,9], size=(3,5)))
    print("choice with probs len=10:", np.unique(
        random.choice([3,5,7,9], p=[0.1,0.3,0.6,0.0], size=1000), return_counts=True))

    # shuffle vs permutation
    arr = np.array([1,2,3,4,5])
    random.shuffle(arr)
    print("shuffle in-place:", arr)
    print("permutation new:", random.permutation(np.array([1,2,3,4,5])))


def ufuncs_and_math():
    section("Ufuncs and arithmetic")
    x = [1,2,3,4]; y = [4,5,6,7]
    print("np.add:", np.add(x,y))
    arr1 = np.array([10,11,12,13,14,15])
    arr2 = np.array([1,2,2,3,4,5])
    print("subtract:", np.subtract(arr1, arr2))
    print("multiply:", np.multiply(arr1, arr2))
    print("divide:", np.divide(arr1, arr2))
    print("power:", np.power(arr1, arr2))
    print("mod:", np.mod(arr1, arr2))
    print("remainder:", np.remainder(arr1, arr2))
    print("divmod:", np.divmod(arr1, arr2))
    print("absolute:", np.absolute(arr1))

    # Rounding
    print("trunc:", np.trunc([-3.1666, 3.6667]))
    print("fix:", np.fix([-3.1666, 3.6667]))
    print("around(2):", np.around(3.1666, 2))
    print("floor:", np.floor([-3.1666, 3.6667]))
    print("ceil:", np.ceil([-3.1666, 3.6667]))

    # Logs
    arr = np.arange(1, 10)
    print("log2:", np.log2(arr))
    print("log10:", np.log10(arr))
    print("ln:", np.log(arr))

    # Reductions
    a1 = np.array([1,2,3]); a2 = np.array([1,2,3])
    print("add:", np.add(a1, a2))
    print("sum all:", np.sum([a1, a2]))
    print("sum axis=1:", np.sum([a1, a2], axis=1))
    print("cumsum:", np.cumsum(a1))

    # Products
    arr = np.array([1,2,3,4])
    print("prod:", np.prod(arr))
    print("prod both:", np.prod([a1, a2]))
    print("prod axis=1:", np.prod([a1, a2], axis=1))
    print("cumprod:", np.cumprod(arr))

    # Differences
    arr = np.array([10, 15, 25, 5])
    print("diff n=1:", np.diff(arr))
    print("diff n=2:", np.diff(arr, n=2))

    # Number theory
    print("lcm(4,6):", np.lcm(4,6))
    print("lcm.reduce 1..10:", np.lcm.reduce(np.arange(1,11)))
    print("gcd(6,9):", np.gcd(6,9))
    print("gcd.reduce:", np.gcd.reduce(np.array([20,8,32,36,16])))

    # Trig and hyperbolic
    print("sin(pi/2):", np.sin(np.pi/2))
    print("sin vector:", np.sin(np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])))
    print("deg2rad:", np.deg2rad(np.array([90,180,270,360])))
    print("rad2deg:", np.rad2deg(np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])))
    print("arcsin(1):", np.arcsin(1.0))
    print("arcsin vec:", np.arcsin(np.array([1, -1, 0.1])))
    print("hypot(3,4):", np.hypot(3,4))
    print("sinh(pi/2):", np.sinh(np.pi/2))
    print("cosh vec:", np.cosh(np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])))
    print("arcsinh(1):", np.arcsinh(1.0))
    print("arctanh vec:", np.arctanh(np.array([0.1, 0.2, 0.5])))

    # Sets
    arr = np.array([1,1,1,2,3,4,5,5,6,7])
    print("unique:", np.unique(arr))
    set1 = np.array([1,2,3,4]); set2 = np.array([3,4,5,6])
    print("union1d:", np.union1d(set1, set2))
    print("intersect1d:", np.intersect1d(set1, set2, assume_unique=True))
    print("setdiff1d:", np.setdiff1d(set1, set2, assume_unique=True))
    print("setxor1d:", np.setxor1d(set1, set2, assume_unique=True))


def random_distributions_small():
    """Small, non-plotting samples of common distributions."""
    section("Random distributions (samples)")
    print("normal sample (2x3):\n", random.normal(size=(2,3)))
    print("normal loc=1, scale=2 (2x3):\n", random.normal(loc=1, scale=2, size=(2,3)))
    print("binomial n=10, p=0.5 (size=10):", random.binomial(n=10, p=0.5, size=10))
    print("poisson lam=2 (size=10):", random.poisson(lam=2, size=10))
    print("uniform (2x3):\n", random.uniform(size=(2,3)))
    print("logistic loc=1 scale=2 (2x3):\n", random.logistic(loc=1, scale=2, size=(2,3)))
    print("multinomial n=6 fair die:", random.multinomial(n=6, pvals=[1/6]*6))
    print("exponential scale=2 (2x3):\n", random.exponential(scale=2, size=(2,3)))
    print("chisquare df=2 (2x3):\n", random.chisquare(df=2, size=(2,3)))
    print("rayleigh scale=2 (2x3):\n", random.rayleigh(scale=2, size=(2,3)))
    print("pareto a=2 (2x3):\n", random.pareto(a=2, size=(2,3)))
    z = random.zipf(a=2, size=20)
    print("zipf a=2 (first 20):", z)


def main():
    basics()
    copy_view_shape()
    iterate_and_nditer()
    stack_split()
    search_sort_filter()
    random_sampling()
    ufuncs_and_math()
    random_distributions_small()


if __name__ == "__main__":
    main()




