# NumPy, translated from plain Python

**A syntax-first guide to indexing, vectorisation and broadcasting, built from `lab1_review.ipynb`.**

Most NumPy material explains *what* an operation does. This guide is organised around a different
question, which is the one that actually builds fluency:

> "I already know how to do this with a Python `for` loop / list comprehension. Is there a simpler
> way to *write* it?"

Every concept below is presented in this fixed template:

| Label | Meaning |
|---|---|
| 🐍 **Python native** | how you would write it with lists, loops, `zip`, `sum`, … |
| 🤔 **Surely there's a simpler way?** | the question worth asking about that native code |
| ✅ **NumPy** | the answer, with real output |
| 🧠 **Design notes** | *why* the NumPy syntax looks the way it does (heavily used in the indexing sections) |
| 🏋️ **Advanced problem** | harder than the notebook example, answer hidden in a collapsible block |

**How to use it.** Read §§6–14 slowly if indexing is your weak spot; that is where the
"syntax philosophy" lives (`:`, `,`, `...`, `None`, lists of ints, boolean masks, and the
view-vs-copy law that explains all of them). §§1–5 and 15–20 are the rest of the notebook.
Every code snippet below was executed on **NumPy 2.5.1 / CPython 3.11+**, and the printed
outputs are real, not remembered. The original notebook stores no outputs, so where its inline
`# Output:` comments disagree with a machine, this guide follows the machine and says so.

### Contents

- **Part I — Foundations**: [1. Vectorisation](#1-vectorisation) · [2. Creating arrays](#2-creating-arrays) · [3. Dimensional attributes](#3-dimensional-attributes-shape-ndim-size) · [4. Reshape & transpose](#4-reshape-and-transpose) · [5. Data types](#5-data-types-dtype)
- **Part II — Indexing (the core)**: [6. The grammar of `[...]`](#6-the-grammar-of--what-is-actually-inside-the-brackets) · [7. Slicing](#7-basic-indexing-and-slicing) · [8. The comma](#8-multidimensional-indexing-and-the-comma) · [9. `...`](#9--the-ellipsis) · [10. `None`](#10-none--npnewaxis-inserting-an-axis) · [11. Integer arrays](#11-integer-array-fancy-indexing) · [12. Boolean arrays](#12-boolean-array-indexing) · [13. Mixing + the view/copy law](#13-mixing-styles-and-the-law-that-explains-copies) · [14. Writing through an index](#14-indexing-on-the-left-hand-side-assignment)
- **Part III — Operations**: [15. Arithmetic & matmul](#15-arithmetic-and-matrix-operations) · [16. Reductions & `axis`](#16-reductions-and-what-axis-really-means) · [17. Stacking](#17-stacking-and-concatenating) · [18. Broadcasting](#18-broadcasting)
- **Part IV — Mechanics & tools**: [19. Copies & views in practice](#19-copies-and-views-in-practice) · [20. Random numbers](#20-random-numbers)
- **Appendices**: [A. Index cheat sheet](#appendix-a--index-expression-cheat-sheet) · [B. Shape rules & error messages](#appendix-b--shape-rules-and-the-errors-they-produce) · [C. Three-step reading method](#appendix-c--how-to-read-any-index-expression-in-three-steps) · [D. Gotchas](#appendix-d--gotcha-list)

---

## 0. Two ideas that explain every syntax choice

Before the operations, two facts about what an `ndarray` *is*. Almost every "weird" thing about
NumPy syntax is a direct consequence of them.

### 0.1 An array is a flat buffer + a shape + strides

```
       buffer:  [0][1][2][3][4][5]          (one contiguous blob of memory)
       shape:   (2, 3)
       strides: (24, 8)                    (bytes to jump to move one step along each axis)

       a[i, j]  ==  buffer at byte offset  base + i*24 + j*8
```

The `shape` is what you see; the `strides` are how NumPy gets there. Three consequences:

1. **Slicing moves no data.** `a[1:]` is just the same buffer with a new `offset` and `shape`. So it
   is O(1), it uses no extra memory, and it *shares memory* with `a`.
2. **Because it shares memory, a slice is writable and writes back**: `v = a[0]; v[:] = 0` changes `a`.
3. **Any operation expressible as "new offset + new shape + new strides" is a *view*.** That is
   `reshape`, `transpose`, `a.T`, `a[...]`, `a[1:4, ::2]`, `a[None]`. Operations that need a
   *different element order* (fancy indexing, `flatten()`, `copy()`) must physically move data and
   therefore return a **copy**.

### 0.2 Everything inside `[...]` is *one* expression handed to `__getitem__`

Python translates `obj[x]` into `type(obj).__getitem__(obj, x)`. Everything between the brackets is a
single expression, and its *type* is what decides the behaviour. The snippet below is about the
right-hand column only: it writes out, longhand, the call that each bracket expression is sugar for.

```python
a = np.arange(12).reshape(3, 4)     # any array, so that every line below is legal

a[2]        #  2 is an int                        -> a.__getitem__(2)
a[1:4]      #  sugar for slice(1, 4)              -> a.__getitem__(slice(1, 4))
a[0, 1]     #  0, 1 is a *tuple* (0, 1)          -> a.__getitem__((0, 1))
a[[0, 2]]   #  a list of ints                    -> a.__getitem__([0, 2])
a[a > 3]    #  a boolean array                   -> a.__getitem__(<bool array>)
```

This single mechanism is the whole story of NumPy indexing, and it is also why the syntax can be
extended without new keywords. A plain `list` refuses the tuple form
(`TypeError: list indices must be integers or slices, not tuple`); `ndarray` accepts *any* of these
objects as a meaningful "index", so `a[0, 1]` can mean "axis 0 = 0 and axis 1 = 1" instead of two
separate `a[0][1]` lookups.

---

# Part I — Foundations

## 1. Vectorisation

*notebook §1*

> 🐍 **Python native** — the running example is a sum of products (a dot product):
> ```python
> x = [1, 2, 3, 4]
> y = [10, 20, 30, 40]
>
> acc = 0
> for a, b in zip(x, y):
>     acc += a * b          # 300
> ```

> 🤔 **Surely there's a simpler way?** — In pure Python the loop can be shortened to
> `sum(a * b for a, b in zip(x, y))`, or, if you reach for the standard library,
> `sum(map(operator.mul, x, y))`. But notice what none of those can give you:
> `x * y` is *not* element-wise multiplication for lists — `[1, 2] * 2` is `[1, 2, 1, 2]`
> (repetition!), and `[1, 2] * [3, 4]` raises `TypeError`. Python has no element-wise operator, so
> the loop is unavoidable at the level of the language. That missing operator **is** the reason
> NumPy exists.

> ✅ **NumPy**
> ```python
> x = np.array([1, 2, 3, 4])
> y = np.array([10, 20, 30, 40])
>
> x * y          # array([10, 40, 90, 160])   <- one expression, no loop
> (x * y).sum()  # 300
> x @ y          # 300                         <- dot product, even shorter
> ```

The second benefit is speed: the loop runs in the interpreter, one Python object per step, while
`x * y` runs one pre-compiled C loop over raw bytes. In the notebook's benchmark
(`arange(1_000_000)`, sum of products) the ratio on a normal laptop is roughly 100×–400×.

> 🧠 **Design notes**
> - **Backwards compatible, not a new language.** `x * y` works because `ndarray`
>   implements `__mul__`; nothing was added to Python's grammar.
> - **Element-wise by default, matrix ops on demand.** `*` is element-wise (Hadamard) and `@` is
>   matrix multiplication. Big deal in the notebook's §7.2 and a classic source of shape bugs.
> - **The general rule of thumb:** if you typed `for` over elements, ask whether the loop can be
>   expressed as an operation on whole arrays. If yes, NumPy is shorter *and* faster.

> 🏋️ **Advanced problem**
> Given `x` and `y` of the same length, return the sum of the products **only over positions where
> both values are positive**, and the mean of those products — without any Python loop.

<details><summary>Solution</summary>

```python
x = np.array([ 3, -1,  4,  2, -5,  6])
y = np.array([ 2,  7, -3,  5,  8, -1])

mask = (x > 0) & (y > 0)     # boolean array, one True/False per position
p    = x[mask] * y[mask]     # mask *and* multiply, still no loop
p.sum(), p.mean()            # (np.int64(16), np.float64(8.0))
```

`x > 0` is a *vectorised comparison* producing a boolean array; `&` combines two of them
(§12 explains why it is `&` and not `and`). Equivalent alternative:
`p = np.where(mask, x * y, 0)`. Note what `p.mean()` divides by: the number of *selected* pairs, not
the size of the original arrays. `p.sum() / mask.sum()` gives the same `8.0`, while
`p.sum() / x.size` gives `2.67` — dividing by the wrong count is the classic masked-average bug.
</details>

---

## 2. Creating arrays

*notebook §3*

> 🐍 **Python native**
> ```python
> l = [[1, 2, 3], [2, 3, 4]]        # "array" as a list of lists
> zeros = [[0] * 3 for _ in range(2)]      # 2 rows x 3 cols
> rng   = list(range(10))                  # [0, 1, ..., 9]
> step  = list(range(2, 10, 2))            # [2, 4, 6, 8]
> grid  = [i / 10 for i in range(11)]      # 11 evenly spaced points in [0, 1]
> ```
> Note the classic trap: `[[0] * 3] * 2` gives two names for **the same** row.

> 🤔 **Surely there's a simpler way?** — The comprehensions are fine, but they (a) build Python
> objects one at a time, (b) produce nested lists rather than a shape, and (c) for the grid, silently
> accumulate floating-point error because every element is computed from scratch by repeated
> addition of `0.1`.

> ✅ **NumPy**
> ```python
> np.array(l)                     # array([[1, 2, 3], [2, 3, 4]])  <-- shape (2, 3)
>
> np.zeros((2, 3))                # all zeros
> np.ones((2, 3))                 # all ones
> np.full((2, 3), 7.5)            # all 7.5          <- the general case
> np.zeros_like(np.ones((2, 3)))  # same shape/dtype as the argument
>
> np.arange(10)                   # array([0, ..., 9])         like range(10)
> np.arange(2, 10, 2)             # array([2, 4, 6, 8])         like range(2, 10, 2)
> np.linspace(start=0, stop=1, num=11)   # 11 points from 0.0 to 1.0 inclusive
> np.logspace(start=1, stop=4, num=4)    # array([10., 100., 1000., 10000.])
> ```

Two creation functions that are worth knowing because they replace common loops:
`np.eye(n)` (identity matrix — `np.eye(4)[labels]` is the one-line one-hot encoder, §11) and
`np.arange(...).reshape(...)` (make then reshape, extremely common).

> 🧠 **Design notes**
> - `np.arange` deliberately mirrors built-in `range`: stop-exclusive, optional step, integers.
> - `np.linspace` deliberately does **not** mirror it: you ask for the *number of points*
>   (`num=`) and you get both endpoints. For floats, prefer it — it computes `start + k*step`
>   instead of accumulating `+0.1` eleven times, so the last point is exactly `1.0`.
> - `logspace(start, stop, num)` is powers of ten: `num=4` over `[1, 4]` gives
>   `10**1, 10**2, 10**3, 10**4`. (In the notebook's phrasing, "start/stop" are *exponents*.)
> - `np.arange(10)` is `int64`, `np.linspace(0, 1, 11)` is `float64`. The dtype follows the
>   arguments, which matters a lot in §5.

> 🏋️ **Advanced problem**
> Using only creation functions and one arithmetic expression each, build: (a) the 5×5
> multiplication table, (b) the 4×4 table `2**i * 3**j`, (c) an 8×8 checkerboard of 0s and 1s.

<details><summary>Solution</summary>

```python
np.arange(1, 6)[:, None] * np.arange(1, 6)
# [[ 1  2  3  4  5]
#  [ 2  4  6  8 10]
#  ...            [25]]

2 ** np.arange(4)[:, None] * 3 ** np.arange(4)

(np.arange(8)[:, None] + np.arange(8)) % 2      # [[0 1 0 1 ...], [1 0 1 0 ...], ...]
```

`[:, None]` inserts a length-1 axis, turning the left operand into a *column vector* and the right
into a *row vector*; the product/mask is then the outer combination. `[None]` is explained properly
in §10, broadcasting in §18. Checkerboard via parity of `i + j` — the same trick as `(i + j) % 2`
in a nested loop, just without the loops.
</details>

---

## 3. Dimensional attributes: `shape`, `ndim`, `size`

*notebook §4*

> 🐍 **Python native**
> ```python
> l = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
>
> len(l)          # 2      outer length          -> shape[0]
> len(l[0])       # 2      length of one middle -> shape[1]
> len(l[0][0])    # 2      length of one inner  -> shape[2]
> ```

> 🤔 **Surely there's a simpler way?** — You cannot ask a nested list for its *shape*; you have to
> hand-measure each level, and the answer is only meaningful if the structure is rectangular.
> `len(l)` is a special case of a more general idea, `shape = (len at each level)`.

> ✅ **NumPy**
> ```python
> arr = np.ones((2, 3, 4))
>
> arr.shape     # (2, 3, 4)   <- a plain tuple
> arr.ndim      # 3           <- len(arr.shape)
> arr.size      # 24          <- math.prod(arr.shape)
> arr.shape[0]  # 2  ... and arr.shape[-1] == 4 (negative indexing works on the tuple too!)
>
> np.array(2).shape   # ()      a 0-d array: ndim 0, size 1
> ```

> 🧠 **Design notes** — `shape` is a **tuple**, not special syntax. That single choice gives you the
> entire Python tuple toolbox for free: comparison `arr.shape == (3, 4)`, membership `1 in arr.shape`,
> slicing `arr.shape[:2]`, unpacking `h, w = img.shape`, `max(arr.shape)`, and `arr.shape[-1]`.
> The mental model "an array *is* its shape tuple plus a buffer" (§0.1) is why `ndim` and `size`
> are *derived* quantities rather than stored ones.

> 🏋️ **Advanced problem**
> Write a one-expression predicate that is `True` iff `arr` has at least one axis of length 1, and
> another that is `True` iff `arr` can be reshaped to any other shape **without copying** (see §19
> for why that is the interesting question, not "can it be reshaped at all").

<details><summary>Solution</summary>

```python
arr = np.ones((2, 3, 4))             # the example from the snippet above

1 in arr.shape                       # (a) any length-1 axis?

arr.flags.c_contiguous               # (b) True  -> reshape is free (a view)
arr.T.flags.c_contiguous             #     False -> reshape has to copy
```

Equivalently, in pure tuple logic (and answering a different question):

```python
[d for d in arr.shape if d != 1]     # the "real" axes; len(...) == 0 for a 0-d array
arr.ndim != arr.squeeze().ndim       # does squeezing this array change anything?
```

Note that *any* array can be reshaped to 2-D — `arr.reshape(2, -1)` succeeds for every non-empty
array, it just copies when the layout does not allow a view. So the real question is always
"free (view) or not (copy)", which `c_contiguous` predicts for the default C order (`order='F'`
reshapes need `f_contiguous` instead).

Both work because `shape` is an ordinary tuple, so ordinary Python logic composes with it. This
matters in real code: `(N, H, W, C)` image tensors are often reshaped with
`x.reshape(x.shape[0], -1)`, which assumes `x.shape[0]` exists and says nothing about the rest.
</details>

---

## 4. Reshape and transpose

*notebook §4.1–4.2*

> 🐍 **Python native** — flattening a matrix without `numpy` is a comprehension, and transposing is
> a double comprehension:
> ```python
> m = [[0, 1, 2], [3, 4, 5]]
>
> flat = [x for row in m for x in row]                    # [0, 1, 2, 3, 4, 5]
> t    = [[row[j] for row in m] for j in range(len(m[0]))]  # [[0, 3], [1, 4], [2, 5]]
> ```

> 🤔 **Surely there's a simpler way?** — The transpose comprehension is O(n) Python-level work and
> its readability collapses at 3 dimensions (a triple comprehension, and four loops for a 4-D
> transpose). Also, you cannot "reshape in place": both snippets allocate new lists.

> ✅ **NumPy**
> ```python
> arr = np.arange(6).reshape(2, 3)   # [[0 1 2]
>                                    #  [3 4 5]]
> arr.reshape(3, 2)                  # 3*2 == 6, reinterprets the same elements
> arr.reshape(6, -1, 1).shape        # (6, 1, 1)   <- -1: "you figure this axis out"
>
> arr.T                              # [[0 3]
>                                    #  [1 4]
>                                    #  [2 5]]
> arr.transpose()                    # same as .T for 2-D
> np.arange(24).reshape(2, 3, 4).transpose(2, 0, 1).shape   # (4, 2, 3)
> ```

> 🧠 **Design notes**
> - **`reshape` preserves the *reading order* of the elements, `transpose` permutes the *axes*.**
>   For `arr = arange(6).reshape(2, 3)`, `arr.reshape(4, 2, 3)` and `arr.transpose(2, 0, 1)` contain
>   different numbers — the notebook makes this point explicitly, and it is the single most common
>   confusion between the two.
> - **`.T` reverses the axis order; `transpose(*axes)` chooses one.** For 2-D, both are the same.
>   `transpose(2, 0, 1)` on a `(2, 3, 4)` array reads "new axis 0 was old axis 2, new axis 1 was old
>   axis 0, new axis 2 was old axis 1" → `(4, 2, 3)`.
> - **`-1` in `reshape` means "infer this one"**: it is a convenience *only*, and at most one `-1`
>   is allowed. `arr.reshape(3, -1, 1)` is `(3, 8, 1)` because `24 / 3 / 1 = 8`.
> - **Neither mutates.** Both *return* a new object. In `reshape`'s case the object may be a view
>   (§19), so `arr.shape` is unchanged but memory is shared.
> - The 1-D gotcha: `np.arange(4).T` is still `(4,)`. There is nothing to transpose; you need
>   `a[:, None]` (§10) to get a column.

> 🏋️ **Advanced problem**
> (a) Predict `np.zeros((8, 2, 5, 3)).transpose(1, 0, 3, 2).shape` (notebook §4.2 exercise).
> (b) Rotate a matrix 90° counter-clockwise **and** clockwise using only `.T` plus slicing.
> (c) Convert a channels-last image batch `imgs` of shape `(N, H, W, C)` to channels-first
> `(N, C, H, W)`.

<details><summary>Solution</summary>

```python
# (a) new axes come from old axes 1, 0, 3, 2
np.zeros((8, 2, 5, 3)).transpose(1, 0, 3, 2).shape    # (2, 8, 3, 5)

# (b) counter-clockwise and clockwise
m = np.arange(6).reshape(2, 3)
np.rot90(m)          # == m.T[::-1]        <- reverse the ROWS of the transpose
np.rot90(m, -1)      # == m[::-1].T        <- transpose of the reversed rows

# (c)
imgs = np.zeros((8, 32, 32, 3))                      # (N, H, W, C)
imgs.transpose(0, 3, 1, 2).shape                     # (N, C, H, W)
```

Part (b) is the payoff of the "buffer + strides" model: a rotation is not a data movement, it is
"reverse this axis and swap the other two", i.e. a negative stride plus a permutation. It stays a
view, so it is free even for huge matrices. Part (c) is the everyday version of the same idea, and
the argument list reads as "new axis k is old axis `axes[k]`".
</details>

---

## 5. Data types (`dtype`)

*notebook §5*

> 🐍 **Python native**
> ```python
> l = [1, 2, 3.14]        # one list, three different object types; nobody enforces anything
> l[0] = "hello"          # perfectly fine, the list just changes
> ```
> In CPython every number is a `PyObject*` with a header, a type pointer and a reference count —
> the same `int` costs ~28 bytes.

> 🤔 **Surely there's a simpler way?** — For a 1000×1000 grid of numbers, a list-of-lists has a
> million boxed objects plus a million pointers. There is no way to say "this container holds
> 64-bit floats, unboxed" in pure Python, and therefore no way to do arithmetic faster than one
> object at a time.

> ✅ **NumPy**
> ```python
> np.array([1, 2, 3]).dtype          # dtype('int64')      on Linux/macOS; int32 on Windows
> np.array([1, 2, 3.14]).dtype       # dtype('float64')    ints widened: one dtype per array
> np.arange(10, dtype=np.float32).dtype   # dtype('float32')  <- at creation
> np.arange(10).astype(np.float32)        # .astype(...)      <- after creation (returns a copy)
> (np.arange(10) * 3.14).dtype       # dtype('float64')    <- implicit promotion
>
> bools = np.array([True, False, True])
> bools * 1                          # array([1, 0, 1])    bool -> int, and back via .astype(bool)
> ```

> 🧠 **Design notes**
> - **One dtype per array is the price of speed.** Unboxed memory can be walked with a single C loop,
>   but every element must have the same size. Hence "less general types are stored in more general
>   types" — the notebook's §5 rule.
> - **Assignment casts into the existing dtype, silently**: `a = np.array([1, 2, 3]); a[0] = 1.99`
>   leaves `a == array([1, 2, 3])`. The `1.99` was truncated to `1`, not rounded, and the array's
>   dtype was *not* upgraded. If the conversion is impossible (`a[0] = 'COMP2211'`) you get
>   `ValueError: invalid literal for int() with base 10: 'COMP2211'`.
> - `astype` always copies; `astype` narrowing truncates toward zero (`np.float64(2.9).astype(int)`
>   is `2`), while `np.round(x).astype(int)` rounds first.
> - Integer overflow is silent-by-default inside `int64`, and wraps for narrow types:
>   `np.int8(127) + np.int8(1)` is `-128` (NumPy 2 at least emits a `RuntimeWarning` here).

> 🏋️ **Advanced problem**
> (a) Why does `a[0] = 1.99` store `1`, and how would you store `2`?
> (b) A dataset of 1e6 `float64` values takes how much memory as `float64` vs `float32`?
> (c) What does `np.array([1, 2, 3]).astype(bool)` give, and what does
> `np.array([0, 0.5]).astype(bool)` give? Why the difference?

<details><summary>Solution</summary>

```python
# (a) truncation toward zero, because the target dtype is already int64
a = np.array([1, 2, 3]); a[0] = 1.99          # 1
np.round(1.99).astype(a.dtype)                # 2

# (b) 1e6 * 8 bytes = 8 MB  vs  1e6 * 4 bytes = 4 MB
np.zeros(10**6, np.float64).nbytes / 1e6      # 8.0
np.zeros(10**6, np.float32).nbytes / 1e6      # 4.0

# (c)
np.array([1, 2, 3]).astype(bool)   # array([True, True, True])    any non-zero is True
np.array([0, 0.5]).astype(bool)    # array([False, True])         only 0 is False
```

`bool` is a real dtype with 1 byte per element, and `a == 0` / `a != 0` are the idiomatic ways to
get it; `.astype(bool)` is mostly used to *store* a mask or to convert a mask into memory you can
write into (§12 shows `a * mask`, `a[mask]` and `a[mask] = 0`, which are three different things).
</details>

---

# Part II — Indexing: the core

This is the part that answers your actual question: **what do `[:]`, `[,]` and `[...]` mean, and why
did NumPy choose those spellings?** Read §6 first; the rest are applications of it.

## 6. The grammar of `[...]`: what is actually inside the brackets

*notebook §6*

> 🐍 **Python native**
> ```python
> l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
> l[1]        # 1            single element
> l[-1]       # 9            from the end
> l[1:4]      # [1, 2, 3]    a sub-list
> l[4::2]     # [4, 6, 8]
> l[::-1]     # [9, ..., 0]  reversed
> l[0][1]     # TypeError-ish: l[0] is an int, not a container
> ```
> Everything on the right of `l[` and left of `]` is one Python expression. Python does not have a
> special grammar for "index": it calls `l.__getitem__(that_expression)`.

> 🤔 **Surely there's a simpler way?** — Two things are missing from list indexing. First, the colon in
> `l[1:4]` looks like special syntax but is actually *sugar*, and knowing that unlocks everything.
> Second, there is no way to address "element `i` of level 1, element `j` of level 2" in one
> expression — you must chain `l[i][j]`, which is the thing that does not generalise to N dimensions.

> ✅ **NumPy**
> ```python
> a = np.arange(10)
>
> a[1]              # np.int64(1)         <- identical to list indexing
> a[1:4]            # array([1, 2, 3])    <- right end EXCLUDED
> a[1:4]            # is exactly the same call as:
> a[slice(1, 4)]    # array([1, 2, 3])
> a[1:4, ]          # array([1, 2, 3])    <- trailing comma: a 1-tuple, harmless
> type(slice(1, 4)) # <class 'slice'>
> ```

**So `:` is not an operator — it is a constructor for `slice` objects**, and `[x:y:z]` is sugar for
`[slice(x, y, z)]`. The complete menu of what can appear inside `[...]`:

| Inside `[...]` | Python type | Name | What it means |
|---|---|---|---|
| `2`, `-1` | `int` | basic | pick **one** position on this axis, **drop** the axis |
| `1:4`, `:`, `::2`, `::-1` | `slice` | basic | pick a **range** on this axis, **keep** the axis |
| `...` | `Ellipsis` | basic | "`:` for every axis I did not write" |
| `None` | `NoneType` | basic | **insert** a new length-1 axis here (does not consume an axis) |
| `[0, 2, 3]`, `np.array([0, 2])` | `list`/`ndarray` of ints | advanced | **gather** in that order, axis is consumed, result is a **copy** |
| `a % 3 == 0` | `ndarray` of bools | advanced | **mask**: take the `True` positions, result is a **copy** |
| `(0, 1)` | `tuple` | — | the separator! one entry **per axis**, written as `a[0, 1]` |

The comma-separated entries are *per-axis addresses*, and `:` is the **wildcard address** for an axis:
"no constraint on this axis". That is the whole meaning of `[:]`, `[:, ]`, `[:, 1]`, `[..., 0]`:

- `a[:]` → one address, and it is "all of axis 0". The array you get back **is a view of the entire
  array** (not a copy — contrast with `l[:]` for lists, which *does* copy).
- `a[:, 1]` → two addresses: "all of axis 0, position 1 on axis 1".
- `a[..., -1]` → "all axes I did not mention, then position -1 on the last axis".

> 🧠 **Design notes (the philosophy you asked for)**
> 1. **Why reuse `:` at all?** Because Python already had it for sequences, so `a[1:4]` needs no new
>    syntax and no relearning. NumPy extended the *meaning* by letting the thing inside `[...]` be
>    richer than an `int` — a `slice`, a `tuple`, an `ndarray` — while Python's grammar stayed put.
> 2. **Why the comma instead of `a[0][1]`?** `a[0][1]` is two separate address computations: it
>    materialises row 0 (a view), then indexes that view. `a[0, 1]` is *one* address computation over
>    both axes, so NumPy can compute a single offset **and** can see the whole picture (which is what
>    makes "advanced indexing over several axes at once" possible at all, e.g.
>    `a[[0, 2], [1, 3]]`). It also tells NumPy the full intent when *writing*, which is why
>    `a[0, 1] = 5` and `a[0][1] = 5` both work but `a[[0, 2], [1, 3]] = 5` is the only spelling of the
>    scatter version.
> 3. **Why is indexing also assignment syntax?** Because `obj[i] = v` is `obj.__setitem__(i, v)` —
>    the *same* index object is handed over. So any address you can compute, you can also write into.
>    Indexing is not "reading"; it is **address arithmetic**, and an address works in both directions:
>    `a[1:, ::2] = 0`, `a[a < 0] = 0`, `a[[0, 2], 1] = -1`.
> 4. **Why do some index styles copy and others not?** Because an address that is "offset + strides"
>    can be handed back for free (a **view**), while an address that is "in this arbitrary order,
>    with repeats" has to be materialised (a **copy**). §13 makes this a one-line law.
> 5. **Storing an index expression.** `np.s_[1:, ::2]` builds the same tuple you would type inside the
>    brackets, so you can pass it around: `idx = np.s_[1:, ::2]; a[idx]`. Handy when the index
>    expression itself is a parameter of your function.
> 6. **Iteration matches the rows.** `for row in a` walks axis 0, exactly like `for row in l` walks a
>    list of rows, and `len(a)` is `a.shape[0]`. Indexing and iteration therefore read the same way
>    for a 2-D array and a list of lists.

> 🏋️ **Advanced problem**
> Without running it, write down (i) the shape of `a[1:4, ]` when `a` is 1-D, (ii) the difference
> between `a[1:4]` and `a[1:4, ...]` for a 3-D `a`, and (iii) why `a[0, 1]` is valid while `[0, 1]`
> by itself is not an index for a list.

<details><summary>Solution</summary>

```python
a = np.arange(10)
a[1:4, ].shape          # (3,)    a 1-tuple of one slice == a[1:4]

x = np.zeros((2, 3, 4))
x[1:2].shape            # (1, 3, 4)      one slice for axis 0; the rest default to ':'
x[1:2, ...].shape       # (1, 3, 4)      identical, "and everything after axis 0"
```

`a[0, 1]` works because the object `(0, 1)` — a tuple — is passed to `ndarray.__getitem__`, which
interprets a tuple as "one address per axis". A `list` rejects tuples outright
(`TypeError: list indices must be integers or slices, not tuple`), because `list.__getitem__`
only accepts ints and slices. The tuple was always legal Python; it just had no meaning until a
type chose to give it one.
</details>

---

## 7. Basic indexing and slicing

*notebook §6.1*

> 🐍 **Python native**
> ```python
> a = list(range(10))
> a[1], a[-1], a[-2]     # (1, 9, 8)
> a[1:4]                 # [1, 2, 3]
> a[:4]                  # [0, 1, 2, 3]      == a[0:4]
> a[4:]                  # [4, ..., 9]       == a[4:10]
> a[4::2]                # [4, 6, 8]
> a[8:2:-1]              # [8, 7, 6, 5, 4, 3]
> a[:]                   # a COPY of the whole list
> ```

> 🤔 **Surely there's a simpler way?** — List slicing is already short, so the numpy version is
> *the same syntax*. The real question is not "is it shorter", it is **"what did I just get back?"**
> Three differences matter, and they are the source of most beginner bugs:
> `a[1]` vs `a[1:2]` (scalar vs array), `a[:]` (copy of a list, view of an array), and out-of-range
> indices (error for `a[10]`, silently clamped for `a[5:100]`).

> ✅ **NumPy**
> ```python
> a = np.arange(10)
>
> a[1], a[-1], a[-2]     # (np.int64(1), np.int64(9), np.int64(8))   scalars
> a[1:4]                 # array([1, 2, 3])     right end EXCLUDED
> a[:4], a[4:]           # array([0,1,2,3]) / array([4,...,9])   omitted = "to the edge"
> a[4::2]                # array([4, 6, 8])     start=4, stop=<end>, step=2
> a[8:2:-1]              # array([8, 7, 6, 5, 4, 3])   step<0 means walk backwards
> a[:]                   # array([0..9])        a VIEW, not a copy!
>
> a[1]      # np.int64(1)      ndim 0, shape ()      <- element
> a[1:2]    # array([1])       ndim 1, shape (1,)    <- array of one element
> a[10]     # IndexError: index 10 is out of bounds for axis 0 with size 10
> a[5:100]  # array([5, 6, 7, 8, 9])   slices are CLAMPED, no error
> a[5:2]    # array([], dtype=int64)  empty, no error
> ```

The full rule for a slice `start:stop:step`: a **negative step means the walk runs backwards**, so
the defaults flip — `start` defaults to the last element and `stop` to "before the first". That is
why `a[::-1]` is a full reversal rather than an empty slice.

> 🧠 **Design notes**
> - **`a[1]` drops the axis, `a[1:2]` keeps it.** For a `(5, 10)` array this is the difference between
>   `(10,)` and `(1, 10)`. It is not an accident: an integer address *is* a point, a slice is a
>   *range*, and ranges can have any length including 1. The same rule is what makes
>   `a[:, 1]` a 1-D array and `a[:, 1:2]` a 2-D column (§8).
> - **Clamping is deliberate.** Slicing asks for "as much of the range as exists", so `a[5:100]`
>   being the tail is more useful than an exception. Integer indexing asks for a *specific* element,
>   so `a[10]` must fail.
> - **`a[:]` on a list copies, on an array does not.** For NumPy the goal is that slicing be free
>   (§0.1), and a free slice cannot be a copy. This is the first place where the "same syntax,
>   different meaning" rule bites, and §13/§19 give you the tools to check.

> 🏋️ **Advanced problem**
> With `arr = np.arange(0, 50).reshape(5, 10)`, select in one expression each of:
> (a) the block `[[13, 14, 15], [23, 24, 25], [33, 34, 35]]` (notebook §6.2 exercise);
> (b) the bottom-right 3×3 block using **negative** indices;
> (c) the 5th column **while keeping the axis** (result shape `(5, 1)`);
> (d) every other element of the *reversed* flattened array;
> (e) the border (first/last row and first/last column) — explain why no single slice can do this.

<details><summary>Solution</summary>

```python
arr = np.arange(0, 50).reshape(5, 10)

arr[1:4, 3:6]          # (a) rows 1-3, cols 3-5   -> 3x3
arr[-3:, -3:]          # (b) uses negative indices, no len() needed
arr[:, 4:5]            # (c) a SLICE keeps the axis -> (5, 1); arr[:, 4] -> (5,)
arr[::-1].ravel()[::2] # (d) reverse then every second (needs a linearisation)

# (e) border: four slices, because a slice can only describe a rectangle
np.r_[arr[0], arr[1:-1, -1], arr[-1][::-1], arr[1:-1, 0][::-1]]
```

Part (e) is the key insight for the whole of Part II: **basic slicing produces a rectangle in index
space** — one contiguous start:stop range per axis, and that is all. The border is not a rectangle,
so you must either concatenate pieces (`np.r_`, §17) or use integer/boolean indexing (§11–§12),
which are the general-purpose tools for non-rectangular selections. Also note `arr[-1][::-1]` in (e):
the second `[...]` indexes a 1-D *view* of the last row, which is why it works without surprises.
</details>

---

## 8. Multidimensional indexing: the comma

*notebook §6.2*

> 🐍 **Python native**
> ```python
> m = [[0, 1, 2], [3, 4, 5]]
>
> m[1][0]                        # 3        two lookups: row, then element
> rows_1_3_cols_0_2 = [[row[j] for j in range(0, 3)] for row in m[1:4]]
> ```
> For 3-D you write `m[i][j][k]`, and for the *slice of a slice* pattern you write a nested
> comprehension with one `for` per dimension.

> 🤔 **Surely there's a simpler way?** — The chained `[][]` form works, but it collapses badly:
> selecting "rows 1–3, columns 0–2" needs a nested comprehension, and writing to a region
> (`m[1:4][0:3] = ...`) does not mean what you want because `m[1:4]` is already a new list.
> You want to describe **all axes at once**.

> ✅ **NumPy**
> ```python
> a = np.arange(24).reshape(2, 3, 4)      # 3-D: 2 "layers", 3 rows, 4 columns
>
> a[1, 1:3, ::-1]     # axis0 = 1, axis1 = rows 1..2, axis2 = all, reversed
> # array([[19, 18, 17, 16],
> #        [23, 22, 21, 20]])
>
> a[1]                # == a[1, :, :] == a[1, ...]   -> shape (3, 4)
> a[..., ::-1]        # reverse the LAST axis, whatever the ndim is
>
> arr = np.arange(0, 50).reshape(5, 10)
> arr[:, 4]           # shape (5,)     column 4, axis dropped
> arr[:, 4:5]         # shape (5, 1)   column 4, axis kept
> ```

Reading `a[1, 1:3, ::-1]` aloud: *"on axis 0 take position 1; on axis 1 take positions 1 and 2; on
axis 2 take everything, walking backwards."* The order of the comma-separated entries follows the
order of `a.shape`, i.e. rows before columns for a 2-D array.

> 🧠 **Design notes**
> - **`a[0, 1]` is not a faster spelling of `a[0][1]`, it is a different operation.** `a[0]` creates a
>   view of row 0 and then indexes it; `a[0, 1]` is a single address computation. The values agree,
>   and both can be assigned through — *as long as the first step returned a view*. Replace it with
>   fancy indexing and algebra breaks:
>   `a[[0]][1] = 99` writes into a temporary copy and **loses the write** (§13).
> - **The tuple entry count must not exceed `ndim`** (`a[0, 1, 2]` on a 2-D array raises
>   `IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed`), but it
>   *may* be shorter — the rest default to `:`.
> - **Negative indices work per axis**: `arr[-1, -1]` is the bottom-right element. Negative *indices*
>   are fine where slices would be clamped, because they still name an existing element.
> - This is the reason the notebook says "rows go before columns": the comma order is `shape` order,
>   not a row/column convention invented by NumPy.

> 🏋️ **Advanced problem**
> (a) Predict the shapes of `a[1, 1:3, ::-1]`, `a[1:2, 1:3, ::-1]` and `a[1, ...]` for the
> `(2, 3, 4)` array above, and say which ones dropped axis 0.
> (b) With `arr = np.arange(0, 50).reshape(5, 10)`, produce: the last three columns of each row
> **in reverse order**; the whole array with rows reversed; the whole array with columns reversed;
> the whole array rotated 180°.
> (c) Column 4 is `arr[:, 4]` with shape `(5,)`. Without slicing, how do you turn the *existing*
> array into a `(5, 1)` version — and why does `arr[:, 4]` not already have that shape?

<details><summary>Solution</summary>

```python
a = np.arange(24).reshape(2, 3, 4)
a[1, 1:3, ::-1].shape    # (2, 4)     axis 0 dropped (int index)
a[1:2, 1:3, ::-1].shape  # (1, 2, 4)  axis 0 kept (slice index)
a[1, ...].shape          # (3, 4)     axis 0 dropped

arr = np.arange(0, 50).reshape(5, 10)
arr[:, -1:-4:-1]         # [[9, 8, 7], [19, 18, 17], ...]   start=-1, stop=-4 (exclusive), step=-1
arr[::-1]                # rows reversed
arr[:, ::-1]             # columns reversed
arr[::-1, ::-1]          # both -> 180° rotation

# (c) the column already has the right numbers; the "missing" axis is a shape thing, not a data thing
arr[:, 4:5]              # (5, 1) by keeping the axis as a length-1 slice
arr[:, 4][:, None]       # (5, 1) by adding the axis back afterwards
arr[:, 4].reshape(-1, 1) # (5, 1) by reshaping a view
```

All three fixes in (c) are views, so they cost nothing — the difference between `(5,)` and `(5, 1)`
is purely a matter of *strides and shape*, exactly as in §0.1. This matters constantly in
broadcasting: `arr - arr.mean(axis=1)` fails for a `(5, 10)` array (shapes `(5,10)` and `(5,)`), while
`arr - arr.mean(axis=1, keepdims=True)` works (§16).
</details>

---

## 9. `...` (the Ellipsis)

*notebook §6.2*

> 🐍 **Python native** — there is no analogue. To get "everything except the last axis" you write
> `x[:, :, :, :, -1]`, hand-counting the colons for the ndim you happen to have.

> 🤔 **Surely there's a simpler way?** — Counting colons is exactly the kind of thing that breaks the
> moment the input becomes a batch of images instead of one image. What if you could say "all the
> other axes, whichever they are"?

> ✅ **NumPy**
> ```python
> x = np.arange(24).reshape(2, 3, 4)
>
> x[..., -1]          # take the last position on the LAST axis, all others wildcarded
> x[0, ...]           # == x[0] == x[0, :, :]
> x[...]              # == x[:] == the whole array (a view)
>
> def last(x):        # works for (A,), (A, B), (A, B, C), ...
>     return x[..., -1]
> last(np.arange(5))          # 4
> last(np.zeros((2, 3)))      # shape (2,)
> last(np.zeros((2, 3, 4)))   # shape (2, 3)
> ```

> 🧠 **Design notes**
> - `...` is the built-in singleton `Ellipsis`, i.e. the object you get from `...` anywhere in Python.
>   NumPy gives it a meaning inside `[...]`: **"expand to as many `:` as needed"**. `x[..., 0]` on a
>   4-D array means `x[:, :, :, 0]`. It may appear at most once (its expansion would otherwise be
>   ambiguous).
> - **It is the tool for writing ndim-agnostic code.** In a function whose input might be
>   `(N, H, W, C)` today and `(H, W, C)` tomorrow, `x[..., 0]`, `x[..., ::-1]` and `x[..., None]` keep
>   working; `x[:, :, 0]` does not. Compare with `axis=-1` in reductions (§16) and `swapaxes(-1, -2)`:
>   negative indices and `...` are how NumPy lets you stay silent about ndim.
> - `...` is "basic indexing" (it only expands to slices), so it still produces views and can be an
>   assignment target: `x[..., 0] = 0` zeroes one channel of every sample.

> 🏋️ **Advanced problem**
> For an image batch `imgs` of shape `(N, H, W, C)`:
> (a) extract the red channel of every image and state the resulting shape;
> (b) flip every image left-right without touching batch/channel axes;
> (c) append an alpha channel of ones to each image;
> (d) write one expression that works for both `(N, H, W, C)` and `(H, W, C)` inputs for (b).

<details><summary>Solution</summary>

```python
imgs = np.zeros((8, 32, 32, 3))

imgs[..., 0].shape            # (8, 32, 32)          (a)
imgs[..., ::-1, :].shape      # (8, 32, 32, 3)       (b) reverses W only
np.concatenate([imgs, np.ones(imgs.shape[:-1] + (1,))], axis=-1).shape   # (8,32,32,4)  (c)
imgs[..., ::-1, :]            # (d) same expression works for (H, W, C)
```

Part (c) is the shape-building habit worth learning: `imgs.shape[:-1] + (1,)` reads "everything
except the channel axis, then one channel" — tuple slicing plus tuple concatenation, the §3 idea
that `shape` is just a tuple. `np.ones(X.shape)`-style code that plans shapes on the fly is the
NumPy equivalent of type annotations.
</details>

---

## 10. `None` / `np.newaxis`: inserting an axis

*notebook §7.5*

> 🐍 **Python native**
> ```python
> a = [0, 1, 2, 3]
> row    = [a]                 # [[0, 1, 2, 3]]        a 1x4 "matrix"
> column = [[x] for x in a]    # [[0], [1], [2], [3]]  a 4x1 "matrix"
> ```
> Getting a column vector is a comprehension; making it *stay* aligned when you do arithmetic with it
> is left entirely to you.

> 🤔 **Surely there's a simpler way?** — Wrapping each element in a list is ugly and slow, and the
> nesting level you need depends on the ndim you are starting from.

> ✅ **NumPy**
> ```python
> a = np.array([0, 1, 2, 3])          # shape (4,)
>
> a[None, :].shape        # (1, 4)    row vector
> a[:, None].shape        # (4, 1)    column vector
> a[None, :, None].shape  # (1, 4, 1)
> a[..., None].shape      # (4, 1)    "append a trailing axis", ndim-agnostic
>
> np.expand_dims(a, axis=0).shape          # (1, 4)
> np.expand_dims(a, axis=1).shape          # (4, 1)
> np.expand_dims(a, axis=(0, 1, 2)).shape  # (1, 1, 1, 4)
> np.expand_dims(a, axis=(0, 1, 3)).shape  # (1, 1, 4, 1)
>
> a.reshape(1, 4).shape   # (1, 4)  — the same thing, written differently
> ```

> 🧠 **Design notes**
> - **`None` is not a selector.** `:` says "take all of an existing axis"; `None` says "there is a new
>   axis here, of length 1". That is why `a[None]` *increases* `ndim` while `a[:]` does not.
> - **`None` is `np.newaxis`.** `np.newaxis` is literally defined as `None`, so
>   `a[:, np.newaxis]` and `a[:, None]` are the same character-level trick with a friendlier name.
> - **Position matters.** `a[:, None]` makes a column, `a[None, :]` makes a row. Because the new axis
>   has length 1, broadcasting (§18) will stretch it, so where you insert it *is* the meaning of your
>   computation: `a[:, None] + b` vs `a[None, :] + b` are different (and the second one errors when
>   the lengths disagree).
> - It is basic indexing (no data is touched), so it is free and always a view.

> 🏋️ **Advanced problem**
> Using `None` and no loops, build:
> (a) the outer product of `[0, 1, 2, 3]` and `[1, 2]`;
> (b) the one-hot encoding of `np.array([0, 3, 1])` for 4 classes;
> (c) the 3×3 matrix of absolute differences `|x_i - x_j|` for `x = [0, 1, 2]`; then the same thing
> as squared distances for a `(n, d)` matrix `X`.

<details><summary>Solution</summary>

```python
a = np.array([0, 1, 2, 3]); b = np.array([1, 2])

a[:, None] * b                      # (4, 1) * (2,) -> (4, 2) outer product   (a)

(np.arange(4)[:, None] == np.array([0, 3, 1])).astype(int)     # (b) compare (4,1) with (3,)
np.eye(4, dtype=int)[[0, 3, 1]]                                #     ... or index an identity

x = np.array([0, 1, 2])
np.abs(x[:, None] - x[None, :])     # (3,1) - (1,3) -> (3,3)                    (c)

X = np.random.default_rng(0).normal(size=(5, 2))
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))    # (5, 5)
```

In (c) the last line is worth studying: `X[:, None, :]` is `(5, 1, 2)` and `X[None, :, :]` is
`(1, 5, 2)`; the broadcast makes `(5, 5, 2)` differences, and `sum(-1)` collapses the feature axis.
`np.expand_dims(X, 1) - np.expand_dims(X, 0)` is the same expression written with the explicit
function; `[..., None]` and `[:, None, :]` are how you control *where* the new axis lands.
</details>

---

## 11. Integer array (fancy) indexing

*notebook §6.3*

> 🐍 **Python native**
> ```python
> a = list(range(10))
> picks = [0, 2, 3, 9]
>
> selected = [a[i] for i in picks]              # [0, 2, 3, 9]
> m = [[0, 1, 2], [3, 4, 5]]
> diag = [m[i][i] for i in range(2)]            # [0, 4]
> ```
> A union of disjoint positions (`[0, 2, 3, 9]`) cannot be a slice at all — Python has no `l[0:2 + 3 + 9]`
> style syntax — so a comprehension is the only native answer.

> 🤔 **Surely there's a simpler way?** — The comprehension works, but it is a Python loop (slow, and
> it cannot be assigned to: `[a[i] for i in picks] = 0` is a syntax error). For the 2-D diagonal it
> needs an index variable that walks two axes at once, which gets worse at higher ndim.

> ✅ **NumPy**
> ```python
> a = np.arange(10)
> a[[0, 2, 3, 9]]        # array([0, 2, 3, 9])   a list of ints is a valid index
> a[np.array([0, 2, 3, 9])]   # identical, either container works
> a[[-1, -2]]            # array([9, 8])        negative values are allowed
> a[[2, 2, 0]]           # array([2, 2, 0])     repeats duplicate, order is respected
> a[0:4]                # == a[[0, 1, 2, 3]]   slicing is a special case of gathering
>
> m = np.arange(16).reshape(4, 4)
> m[[0, 1, 2, 3], [0, 1, 2, 3]]        # array([ 0,  5, 10, 15])   the diagonal  <- PAIRED
> m[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]  # 2x2: [[0, 5], [10, 15]]   the index SHAPE is the result shape
> ```

**The rule that surprises everyone:** when several integer arrays are supplied, they are
**broadcast against each other** and the result has the *broadcast index shape*. They are not an
outer product:

```python
m[[0, 1], [0, 1]]      # shape (2,)  -> pairs (0,0) and (1,1)   = diagonal
m[[0, 1], [0, 1]]      # NOT the 2x2 corner block
m[[0, 1]][:, [0, 1]]   # (2, 2) corner block: index twice, first rows then columns
m[np.ix_([0, 1], [0, 1])]   # (2, 2) the same block, in ONE index expression
m[[0, 1], [0, 1, 2]]   # IndexError: shape mismatch: indexing arrays could not be broadcast
```

> 🧠 **Design notes**
> - **Why is it called "fancy"?** Because it is a *gather*: "give me these elements, in this order,
>   even with repeats". A gather cannot be expressed as offset + strides, so the result is always a
>   **copy** (§13) and the result *shape follows the index*, not the original array.
> - **Why the paired (broadcasting) rule and not an outer product?** Because the paired rule is the
>   strictly more expressive one: an outer product is recovered by `np.ix_` (or by a 2-D index array),
>   whereas a per-row gather (`m[rows, cols]` with matching lengths) — the pattern you need for
>   "the best column per row", `m[np.arange(n), m.argmax(axis=1)]` — is only expressible as pairing.
>   `np.ix_` exists purely to give you the outer-product spelling when you want a *block*.
> - **Negative values are allowed, out-of-range values are not**:
>   `a[[0, 10]]` → `IndexError: index 10 is out of bounds for axis 0 with size 10`. Floats are
>   rejected, and the wording depends on how you wrote them: `a[np.array([1.5, 2.5])]` says
>   `IndexError: arrays used as indices must be of integer (or boolean) type`, while `a[[1.0, 2.0]]`
>   prints the longer "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and
>   integer or boolean arrays are valid indices" message.
> - **Function forms**: `np.take(a, indices, axis=1)` is `a[:, indices]`, and
>   `np.take_along_axis(a, idx, axis=1)` is the paired form for per-row indices when you do not want
>   to write `np.arange(...)` by hand. `np.put` writes values at index positions.
> - If the index arrays are exactly `arange(k)`, the paired rule degenerates to "pick the diagonal",
>   which is why `m[np.arange(4), np.arange(4)]` is the diagonal and not the corner block.

> 🏋️ **Advanced problem**
> With `arr = np.arange(0, 50).reshape(5, 10)`:
> (a) swap the first and last row (notebook §6.3 exercise);
> (b) copy row 1 three times (notebook §6.3 exercise);
> (c) select the "stair" `[[0, 1], [11, 12], [22, 23]]` (notebook §6.3 exercise);
> (d) select the anti-diagonal of a 4×4 matrix;
> (e) for each row of `arr`, select the element at the row's own `argmax` position;
> (f) take the four corner elements of `arr` — first with fancy indexing only, then with `np.ix_`,
> and explain why `arr[[0, -1], [0, -1]]` is **not** the answer.

<details><summary>Solution</summary>

```python
arr[[4, 1, 2, 3, 0]]                       # (a) a permutation of row indices
arr[[1, 1, 1]]                             # (b) repeats duplicate the row
rows = np.arange(3)[:, None]               # (3, 1)
cols = np.arange(3)[:, None] + np.arange(2)   # (3, 2) -> [[0,1],[1,2],[2,3]]
arr[rows, cols]                            # (c) [[0, 1], [11, 12], [22, 23]]

m = np.arange(16).reshape(4, 4)
n = 4
m[np.arange(n), np.arange(n - 1, -1, -1)]  # (d) [3, 6, 9, 12]

arr[np.arange(5), arr.argmax(axis=1)]      # (e) the row-wise maximum  -> [9, 19, 29, 39, 49]

arr[np.ix_([0, -1], [0, -1])]              # (f) [[0, 9], [40, 49]]
arr[[0, -1]][:, [0, -1]]                   # (f) same values, two index operations
arr[[0, -1], [0, -1]]                      # WRONG: pairs -> [0, 49]  (top-left, bottom-right only)
```

Part (e) is the design pattern behind `np.argmax` + gather, and part (f) is the single most useful
thing to internalise about this syntax: **two index arrays of the same length mean "pairs", not
"all combinations"**. If you want all combinations, either index twice (`arr[[0, -1]][:, [0, -1]]`,
which goes through an intermediate copy) or use `np.ix_`, which reshapes the two lists into
`(2, 1)` and `(1, 2)` so that broadcasting produces the full grid.
</details>

---

## 12. Boolean array indexing

*notebook §6.4*

> 🐍 **Python native**
> ```python
> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>
> keeps   = [x for x in a if x % 3 == 0]        # [0, 3, 6, 9]
> matrix  = [[0, 1, 2], [3, 4, 5]]
> flat    = [x for row in matrix for x in row if x % 3 == 0]   # [0, 3]
> ```
> And there is no native way to write "set every element divisible by 3 to -1" — a comprehension
> builds a *new* list; it cannot write back into the old one.

> 🤔 **Surely there's a simpler way?** — The condition has to be spelled out once per dimension, and
> the "select" and "modify" versions are completely different pieces of code. In NumPy both are the
> same object: a boolean array the size of the data.

> ✅ **NumPy**
> ```python
> a = np.arange(9).reshape(3, 3)
>
> a < 2                     # array([[ True,  True, False],
>                           #        [False, False, False],
>                           #        [False, False, False]])   a MASK, not indices
> a[a < 2]                  # array([0, 1])     selection -> 1-D copy
> a[a < 2] = -1             # in-place scatter through the same mask
>
> (a < 2) | (a > 6)         # OR   (elementwise)
> (a < 2) & (a > 6)         # AND  -> all False here
> ~(a < 2)                  # NOT  == a >= 2
> np.logical_or(a < 2, a > 6)   # the function form of `|`
> np.where(a % 2 == 0, a, -a)   # "masked choose": a where the mask holds, -a otherwise
> ```

The four spellings from the notebook's §6.4 exercise, and what each one *is*:

```python
arr % 3                  # values    array([0, 1, 2, 0, 1, ...])   arithmetic, dtype int
arr % 3 == 0             # mask      array([ True, False, False,  True, ...])  dtype bool
arr[arr % 3 == 0]        # copy      array([0, 3, 6, 9, ..., 48])  a 1-D SELECTION
arr[arr % 3 == 0] = -1   # write     the original array is modified in place
```

**Mask shape rules** (this is where beginners get burned):

| Mask shape | Effect | Result |
|---|---|---|
| same shape as the array, e.g. `(3, 3)` | select every `True` element | **1-D** array, flattened in C order |
| matches axis 0 only, e.g. `(3,)` for a `(3, 3)` array | select **rows** | same ndim, reduced along axis 0 |
| anything else | error | `IndexError: boolean index did not match indexed array along axis 0; size of axis is 3 but size of corresponding boolean axis is 4` |

> 🧠 **Design notes**
> - **Why `&`/`|`/`~` instead of `and`/`or`/`not`?** Python does not allow overloading `and`, `or`
>   or `not` — they are language-level control flow that calls `bool()` on their operands. For an
>   array with more than one element, `bool()` is genuinely ambiguous, so NumPy raises
>   `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or
>   a.all()`. NumPy therefore reuses the *bitwise* operators, which Python **does** allow a type to
>   overload, and which are elementwise for ndarrays.
> - **Precedence trap**: `&` binds tighter than comparisons, so `a > 1 & a < 4` is not
>   `(a > 1) & (a < 4)`. It parses as a chained comparison and raises the ambiguity error; but
>   `m1 & m2 == 0` silently means `(m1 & m2) == 0`. **Always parenthesise comparisons.**
> - **A full-shape mask flattens the result.** `a[a > 4]` on a 3×3 array gives a 1-D array of the
>   surviving values; the shape information is gone. If you need the shape, use `np.where` (keeps
>   the shape, fills the gaps) or a 1-D row mask (keeps the trailing axes).
> - **`mask` is data, not an index list.** You can store it, combine it, negate it, and reuse it.
>   `a[mask] = 0` is the operation Python lists simply cannot express.
> - **Useful companions:** `mask.sum()` / `np.count_nonzero(mask)` (how many), `mask.any()` /
>   `mask.all()` (existential / universal), `np.nonzero(mask)` / `np.argwhere(mask)` (coordinates).

> 🏋️ **Advanced problem**
> With `arr = np.arange(0, 50).reshape(5, 10)`:
> (a) build the mask of values divisible by 3, select those values, and set them to `-1` (notebook
> exercise), then explain the difference between the three expressions;
> (b) replace every negative value by `0` in three different ways, one of which does **not** allocate
> a new array;
> (c) count how many elements are both odd and greater than 40;
> (d) keep only the elements that exceed their **own row's** mean — and explain why the result is
> 1-D, then produce a same-shape version;
> (e) keep only the *rows* whose first element is greater than 25;
> (f) find the coordinates of the maximum element.

<details><summary>Solution</summary>

```python
# (a)
arr = np.arange(0, 50).reshape(5, 10)
mask = arr % 3 == 0            # bool array, shape (5, 10)
arr[mask]                      # 1-D copy: [0, 3, 6, ..., 48]   (17 values)
arr[mask] = -1                 # write-through; arr itself changes

# (b)  (parts c-f each start from a fresh arr, so they do not see (a)'s edits)
a = np.array([-2, 0, 3, -1])
np.where(a < 0, 0, a)          # new array                      -> [0, 0, 3, 0]
np.maximum(a, 0)               # new array, cheaper and clearer -> [0, 0, 3, 0]
a[a < 0] = 0                   # in place, no allocation        -> a is now [0, 0, 3, 0]

# (c)
arr = np.arange(0, 50).reshape(5, 10)
np.count_nonzero((arr > 40) & (arr % 2 == 1))    # 5  (values 41, 43, 45, 47, 49)

# (d)
arr[arr > arr.mean(axis=1, keepdims=True)].shape   # (25,)   <- flattened!
np.where(arr > arr.mean(axis=1, keepdims=True), arr, 0)   # same shape (5, 10)

# (e)
arr[arr[:, 0] > 25]            # rows 3 and 4 -> shape (2, 10)   (a 1-D row mask)

# (f)
np.argwhere(arr == arr.max())  # array([[4, 9]])
```

Two things to take away. First, `keepdims=True` in (d) is **not** cosmetic: without it the
comparison is `(5, 10)` against `(5,)`, which is not broadcastable, and NumPy raises
`operands could not be broadcast together with shapes (5,10) (5,)`. Second, the mask in (d) has the
full shape, so `arr[mask]` flattens; (e) uses a *1-D* mask, so the result keeps its trailing axis.
</details>

---

## 13. Mixing styles, and the law that explains copies

*notebook §6.4 + §9*

> 🐍 **Python native** — there is only one behaviour: `l[i]`, `l[a:b]`, `l[:]`, nested lookups are all
> just "get a reference", and `l[:]` and `list(l)` are the idiomatic shallow copies.

> 🤔 **Surely there's a simpler way?** — With NumPy the same-looking notation can hand you either the
> memory itself or a duplicate of it. Rather than memorising a table of exceptions, you want **one
> rule** that predicts which.

> ✅ **NumPy — the law**

> **If the index expression contains *only* ints, slices, `...` and `None`, you get a view.
> If it contains *any* list/array of ints or booleans, you get a copy.**

```python
a = np.arange(6)

np.shares_memory(a, a[1:4])      # True    slice            -> view
np.shares_memory(a, a[..., None])# True    newaxis          -> view
np.shares_memory(a, a[[1, 2]])   # False   integer array    -> COPY
np.shares_memory(a, a[a > 2])    # False   boolean mask     -> COPY

a = np.arange(6); v = a[1:4]; v[0] = 99
a                                # array([ 0, 99,  2,  3,  4,  5])   wrote THROUGH the view

a = np.arange(6); v = a[[1, 2]]; v[0] = 99
a                                # array([0, 1, 2, 3, 4, 5])        UNTOUCHED: v was a copy
```

Mixing is allowed, and the law applies to the whole expression:

```python
m = np.arange(16).reshape(4, 4)
m[[0, -1], :]        # corners of rows 0 and 3: fancy on axis 0, slice on axis 1  -> COPY
m[1:, [0, 2]]        # rows 1..3, columns 0 and 2                                   -> COPY
m[np.ix_([0, -1], [0, -1])]   # 2x2 block, one expression                           -> COPY
```

**The crucial asymmetry:** reading through fancy indexing gives you a copy, but *writing* through
fancy indexing writes into the original array. `a[[1, 2]] = 0` and `a[a > 2] = 0` both mutate `a`.
In other words:

```python
v = a[[1, 2]]        # v is a copy       -> writing to v is invisible in a
a[[1, 2]] = 0        # direct write      -> a is modified
v = a[[1, 2]]; v[:] = 0        # modifies the copy only  -> a unchanged
```

> 🧠 **Design notes**
> - Views exist because copying is expensive and because `a[1:4] = 0` has to work on the real memory.
> - Copies exist for gathers because "these elements in this order, possibly repeated" cannot be
>   described by strides.
> - `__setitem__` with a gather is not a gather at all — it is a **scatter** ("write these values to
>   these positions"), which is why it must reach the original memory. Read = gather = copy;
>   write = scatter = in place.

> 🏋️ **Advanced problem**
> Explain, with evidence, why `a[[0, 1]][:, 0] = 99` does **not** modify `a`, while
> `a[[0, 1], 0] = 99` does. Then: given `v = a[1:]`, which of `v += 1` and `v = v + 1` changes `a`?

<details><summary>Solution</summary>

```python
a = np.arange(16).reshape(4, 4)

b = a.copy(); b[[0, 1]][:, 0] = 99; b[:, 0]      # [ 0,  4,  8, 12]  unchanged
b = a.copy(); b[[0, 1], 0]    = 99; b[:, 0]      # [99, 99,  8, 12]  modified
np.shares_memory(a, a[[0, 1]])                   # False  <- that is the whole reason

v = a[1:]
v += 1          # in-place op on a VIEW -> a's elements 4..15 all increase by 1
v = v + 1       # rebinds the name v to a NEW array -> a unchanged
```

`a[[0, 1]]` evaluates *first* and yields a temporary copy; `[:, 0] = 99` then writes into that
temporary, which is immediately garbage-collected. `a[[0, 1], 0] = 99` is a single `__setitem__`
call with the index tuple `([0, 1], 0)`, so NumPy addresses the original buffer directly. The
practice rule: **if you want to write, put the whole address in one `[...]`** (`a[[0,1], 0] = 99`,
`a[np.ix_(rows, cols)] = 0`) rather than chaining `[...][...]`.
</details>

---

## 14. Indexing on the left-hand side (assignment)

*notebook §7.1*

> 🐍 **Python native** — list slice assignment already exists and is more capable than people expect:
> ```python
> l = list(range(9))
> l[5] = 999                # single element
> l[1::2] = [111, 333, 555, 777]   # slice assignment: lengths must MATCH
> l[1::2] = 0               # TypeError: can only assign an iterable
> ```

> 🤔 **Surely there's a simpler way?** — Two limitations: the replacement must be an iterable of the
> exact same length (no "fill with this scalar"), and there is no conditional assignment
> ("set to -1 wherever divisible by 3") short of rewriting the list.

> ✅ **NumPy**
> ```python
> a = np.arange(9)
> a[5] = 999            # scalar into a point
> a[1::2] = 999         # SCALAR broadcast into a slice  (Python lists: TypeError)
> a[1::2] = [111, 333, 555, 777]   # or a matching sequence
>
> m = np.arange(9).reshape(3, 3)
> m[[0, -1], :] = 999               # whole rows via fancy indexing
> m[1:, 1:] = np.array([[444, 555], [777, 888]])   # a whole block, shapes must match
> m[...] = 0                        # everything
> m[m % 2 == 1] = -1                # conditional, in one line
>
> np.clip(a, 0, 9, out=a)           # write the result back into the same buffer, no new array
> a += 1                            # in-place arithmetic (see the note below)
> ```

> 🧠 **Design notes**
> - **The right-hand side is evaluated first**, then scattered. That is why
>   `a[a % 2 == 0] = a[a % 2 == 0] * 10` is safe even though the mask and the target refer to the same
>   array: the RHS is a fresh copy of the selected values.
> - **`a[...] = 0` and `a[:] = 0` differ only in spelling** for a 1-D array; with `...` you never have
>   to count axes, so it is the safer idiom.
> - **Broadcasting applies to the replacement value** — that is how `a[1::2] = 999` works, and how
>   `m[:, 0:1] = np.array([[1], [2], [3]])` fills a column.
> - **Shape mismatch produces a readable error**:
>   `m[1:, :] = np.arange(4).reshape(1, 4)` (with `m` of shape `(3, 3)`) →
>   `ValueError: could not broadcast input array from shape (1,4) into shape (2,3)`.
> - **`a += 1` mutates in place; `a = a + 1` rebinds.** If `b = a` was taken earlier, `b` sees the
>   first but not the second. Same story for views: `v = a[1:]; v += 1` changes `a`, because `+=` is
>   `__iadd__` and writes into the shared buffer.
> - **The duplicated-index gotcha.** `a[[0, 0]] += 1` adds 1 only **once**, because the gather is
>   read once and the scatter happens once at each position. For accumulating repeats use
>   `np.add.at(a, [0, 0], 1)`.

> 🏋️ **Advanced problem**
> (a) Set the main diagonal **and** the anti-diagonal of a square matrix to `0`.
> (b) Set the border of a matrix to `0` without touching the interior.
> (c) Centre each column of a `(4, 3)` float array **in place** (`a` must still be the same object).
> (d) Given counts and an index array with repeats, accumulate correctly:
> `counts = np.zeros(3); idx = np.array([0, 0, 2]); ` make `counts == [2, 0, 1]`.

<details><summary>Solution</summary>

```python
n = 4
m = np.arange(16, dtype=float).reshape(n, n)

# (a)
np.fill_diagonal(m, 0)                                  # main diagonal, fast and clear
m[np.arange(n), np.arange(n - 1, -1, -1)] = 0           # anti-diagonal, fancy-index target
# or, in one expression each:
m[np.arange(n), np.arange(n)] = 0

# (b)
m[[0, -1], :] = 0
m[:, [0, -1]] = 0

# (c) in place: use -= so the buffer is written, not rebound
a = np.arange(12, dtype=float).reshape(4, 3)
a -= a.mean(axis=0, keepdims=True)      # keepdims makes (1, 3) broadcast with (4, 3)

# (d) repeats must be accumulated with at() (or np.bincount), not +=
counts = np.zeros(3)
idx = np.array([0, 0, 2])
counts[idx] += 1            # WRONG: [1, 0, 1]
np.add.at(counts, idx, 1)   # right: [2, 0, 1]  ... but we already added 1, so use a fresh array
np.bincount(idx, minlength=3)   # the idiomatic one-liner: array([2, 0, 1])
```

Part (d) is a genuine "fancy indexing is a gather/scatter, not a loop" consequence: with a loop,
`for i in idx: counts[i] += 1` accumulates; with a vectorised scatter all writes to the same address
happen "simultaneously". `np.bincount` and `np.add.at` exist precisely for that case —
the first is the fast counting idiom, the second the general accumulating one.
</details>

---

# Part III — Operations

## 15. Arithmetic and matrix operations

*notebook §7.2*

> 🐍 **Python native**
> ```python
> a = [[0, 1, 2], [3, 4, 5]]
> b = [[10, 11, 12], [13, 14, 15]]
>
> summed = [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]     # elementwise
>
> # matrix product, the way the textbook defines it
> c = [[sum(a[i][k] * b[k][j] for k in range(len(b)))
>       for j in range(len(b[0]))] for i in range(len(a))]
> ```
> Note that the elementwise version is *already* an infix-free nested comprehension, and the matrix
> product is four nested loops' worth of indices.

> 🤔 **Surely there's a simpler way?** — The cost is not just lines of code: the comprehension walks a
> `zip` per row and builds Python objects, and the matrix product is O(n³) *interpreter* steps.

> ✅ **NumPy**
> ```python
> a = np.arange(6).reshape(2, 3)        # [[0 1 2]
>                                       #  [3 4 5]]
> b = np.arange(10, 16).reshape(2, 3)   # [[10 11 12]
>                                       #  [13 14 15]]
>
> a + b, a - b, a * b, a / b, a ** 2, a // 2, a % 3     # all elementwise
> a < b                                                 # elementwise -> bool array (feeds §12)
>
> a @ b.T                    # array([[ 35,  44],
>                            #        [134, 170]])   matrix product: (2,3) @ (3,2)
> (a @ b.T == np.dot(a, b.T)).all()   # True: np.dot and @ agree for 2-D
> ```

The shape rule for `@`: `(m, k) @ (k, n) -> (m, n)` — the **inner** dimensions must match and they
disappear; the outer ones survive. `*` instead requires the two shapes to be broadcastable (§18).
For 2-D that is a nice mnemonic: `*` is "same shape, element by element", `@` is "the k is shared".

> 🧠 **Design notes**
> - **`/` always means true division**, even for integers: `np.arange(3) / 2` is
>   `array([0. , 0.5, 1. ])`. Use `//` for floor division and `%` for the remainder (which is why
>   `a % 3 == 0` is the divisibility mask of §12).
> - **Comparison operators return arrays**, not booleans. `a < b` is a mask; if you want one yes/no
>   answer use `.any()` / `.all()` or `np.array_equal`.
> - **`@` also works for stacks of matrices**: `(B, m, k) @ (B, k, n) -> (B, m, n)`. It contracts the
>   last axis of the left operand with the second-to-last of the right, so it generalises `np.dot`
>   (which is why `np.dot` is now usually written `@`).
> - **In-place variants exist for every arithmetic operator**: `a += 1` writes into `a`'s buffer,
>   `a = a + 1` rebinds the name. The distinction matters the moment there are two references or
>   a view (§13, §19).

> 🏋️ **Advanced problem**
> (a) Normalise every row of a `(4, 3)` matrix to unit length, in one expression.
> (b) Build the full cosine-similarity matrix between all rows of a `(n, d)` matrix.
> (c) Multiply a batch of `(B, m, k)` matrices by their `(B, k, n)` counterparts.
> (d) Solve `(XᵀX + λI) w = Xᵀ y` for `w`, in one expression.

<details><summary>Solution</summary>

```python
X = np.arange(12, dtype=float).reshape(4, 3)

X / np.linalg.norm(X, axis=1, keepdims=True)                 # (a) keepdims -> (4,1) broadcasts

Xn = X / np.linalg.norm(X, axis=1, keepdims=True)            # (b)
S  = Xn @ Xn.T                                               # (4, 4), diagonal is exactly 1.0

A = np.random.default_rng(0).normal(size=(8, 3, 4))          # (c)
B = np.random.default_rng(1).normal(size=(8, 4, 5))
(A @ B).shape                                                # (8, 3, 5)

lam = 0.1
n_features = X.shape[1]
y = np.array([1.0, 0.0, -1.0, 2.0])                                  # (n,) targets
w = np.linalg.solve(X.T @ X + lam * np.eye(n_features), X.T @ y)     # (d)
```

Part (b) is the pattern behind a surprising amount of ML code: `@` produces the whole similarity
matrix at once, and `keepdims=True` is what makes the division line up. Part (d) shows the shape
bookkeeping you should always do in your head: `X.T @ X` is `(d, d)`, `lam * np.eye(d)` is `(d, d)`,
`X.T @ y` is `(d,)`, so `np.linalg.solve` returns `(d,)` — the same shape as one row of `X.T`.
</details>

---

## 16. Reductions and what `axis` really means

*notebook §7.3*

> 🐍 **Python native**
> ```python
> a = [[0, 1, 2], [3, 4, 5]]
>
> total      = sum(sum(row) for row in a)          # 15
> column_sum = [sum(row[j] for row in a) for j in range(3)]   # [3, 5, 7]
> row_sum    = [sum(row) for row in a]                        # [3, 12]
> ```

> 🤔 **Surely there's a simpler way?** — Each "which axis do I collapse?" decision changes the shape of
> the comprehension (one loop or two, which index stays in the inner one). For a 5-D array you would
> be writing five-deep comprehensions, and `min`/`max`/`mean`/`std` would each need their own copy of
> that structure.

> ✅ **NumPy**
> ```python
> a = np.arange(6).reshape(2, 3)
>
> a.sum()               # 15          everything
> a.sum(axis=0)         # [3, 5, 7]   collapse axis 0    (column sums)
> a.sum(axis=1)         # [3, 12]     collapse axis 1    (row sums)
> a.sum(axis=-1)        # [3, 12]     negative axis == counting from the end
> a.min(), a.max(), a.mean(), a.std(), a.var(), a.prod()
> a.argmax(), a.argmax(axis=1)      # positions, not values
>
> b = np.zeros((5, 7, 3, 2, 9))
> b.sum(axis=(0, 1)).shape    # (3, 2, 9)   multiple axes at once
> a.sum(axis=1, keepdims=True).shape   # (2, 1)  keep the collapsed axis as length 1
> ```

**The single sentence that makes `axis` click:**

> `axis=k` means "**iterate along axis `k` and add the results up**", i.e. `k` is the axis that
> **disappears** from the shape. `axis=0` walks down the rows and burns the row axis. For a shape tuple
> the rule is simply *delete the axes you listed*:

```
(5, 7, 3, 2, 9).sum(axis=(0, 1))  ->  (3, 2, 9)
            ^^^^^ deleted
```

> 🧠 **Design notes**
> - **`keepdims=True` exists purely for broadcasting.** The collapsed axes come back as length 1, so
>   the result can be combined with the original array again: `a - a.mean(axis=1, keepdims=True)`
>   works, while `a - a.mean(axis=1)` raises a broadcast error (§18). This is the single most useful
>   optional argument in NumPy.
> - **`argmax` counts in flattened order unless you give it an axis.** `a.argmax()` on a `(2, 3)` array
>   returns `5` (a flat index), not a pair of coordinates; `a.argmax(axis=1)` returns one index per
>   row. To recover coordinates use `np.unravel_index(5, a.shape)` (`(1, 2)`), and for a mask of
>   positions use `np.argwhere`.
> - **`np.sum(a)` vs `sum(a)`**: the built-in works on a 1-D array but iterates in Python for 2-D
>   (adding whole rows as arrays — correct, but slow, and with no `axis=` argument at all). Prefer the
>   method or the `np.` function; `a.sum()` is preferred style, and it accepts `out=` for in-place
>   accumulation.
> - **NaNs poison everything**: `np.mean` propagates NaN; there are `np.nanmean`, `np.nansum`, … for
>   skipping them. `np.isnan(a).any()` is the "is my data dirty?" check.
> - **A reduction does not preserve shape, so think of it as an operation that *removes* information.**
>   That is why `mask.mean(axis=0)` is the neat way to get "fraction of `True` per column".

> 🏋️ **Advanced problem**
> (a) Centre each column of a `(4, 3)` array and standardise each column to unit variance.
> (b) Implement a numerically stable row-wise softmax.
> (c) Compute the weighted mean of each column with weights `[1, 2, 3, 4]`.
> (d) For each row of `arr = np.arange(0, 50).reshape(5, 10)`, find the index and the value of the
> maximum element, using reductions only.
> (e) Verify in one expression that every row of a matrix sums to the same value.

<details><summary>Solution</summary>

```python
a = np.arange(12, dtype=float).reshape(4, 3)

a - a.mean(axis=0)                                              # (a) centre columns
(a - a.mean(axis=0)) / a.std(axis=0)                            #     standardise

z = a - a.max(axis=1, keepdims=True)                            # (b) SUBTRACT first: exp() would overflow otherwise
e = np.exp(z)
e / e.sum(axis=1, keepdims=True)                                #     rows sum to 1

np.average(a, axis=0, weights=[1, 2, 3, 4])                     # (c)

arr = np.arange(0, 50).reshape(5, 10)                           # (d)
idx = arr.argmax(axis=1)                                        # [9, 9, 9, 9, 9]
arr[np.arange(5), idx]                                          # the values, via a §11 gather

M = np.tile(np.arange(4.0), (3, 1))                             # (e) three rows of [0, 1, 2, 3]
M.sum(axis=1)                                                   # [6., 6., 6.]  every row sums alike
np.allclose(M.sum(axis=1) - M.sum(axis=1)[0], 0)                # True
np.ptp(M.sum(axis=1)) == 0                                      # True (peak-to-peak)
```

Part (b) is the standard trick and it is a broadcasting story: subtracting a *per-row* constant does
not change the softmax mathematically (it cancels in the ratio) but it keeps `exp` inside
representable range. Part (d) is the reduction + fancy-gather combination that shows up everywhere in
scoring code. In (e), never compare floats with `==`; use `np.allclose` or `np.ptp` (peak-to-peak).
</details>

---

## 17. Stacking and concatenating

*notebook §7.3*

> 🐍 **Python native**
> ```python
> a = [[0, 1, 2], [3, 4, 5]]
> b = [[6, 7, 8], [9, 10, 11]]
>
> vertically = a + b                  # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]  (4x3)
> horizontally = [ra + rb for ra, rb in zip(a, b)]    # [[0, 1, 2, 6, 7, 8], ...]       (2x6)
> ```
> Note `a + b` *concatenates* for lists but *adds* for arrays — the same symbol, two completely
> different meanings. That is a genuine Python/NumPy trap.

> 🤔 **Surely there's a simpler way?** — The horizontal version needs a comprehension that knows both
> operands row by row, and neither version is expressible for 3-D without another nesting level.

> ✅ **NumPy**
> ```python
> a = np.arange(6).reshape(2, 3)
> b = np.arange(10, 16).reshape(2, 3)
>
> np.vstack((a, b))                      # (4, 3)  stack along axis 0
> np.hstack((a, b))                      # (2, 6)  stack along axis 1
> np.concatenate((a, b), axis=0)         # exactly vstack
> np.concatenate((a, b), axis=1)         # exactly hstack
> np.concatenate((a, b), axis=-1)        # last axis, whatever the ndim
>
> np.stack([a, b], axis=0).shape         # (2, 2, 3)   NEW axis, the operands are NOT merged
> np.column_stack((np.arange(3), np.arange(3) * 10))    # [[0, 0], [1, 10], [2, 20]]
> ```

| Function | What it does | Note |
|---|---|---|
| `concatenate((x, y), axis=k)` | joins along an **existing** axis | requires the same ndim; all other axes must match |
| `vstack` / `hstack` | joins along axis 0 / axis 1 | the 1-D special cases surprise people (below) |
| `stack((x, y), axis=k)` | creates a **new** axis | all inputs must have identical shapes |
| `column_stack` | turns 1-D arrays into columns | the fast way to build a feature matrix |
| `np.append` | "concatenate, but worse" | allocates a copy every call; never use in a loop |

The 1-D trap: `np.vstack((a, b))` with two `(3,)` arrays gives `(2, 3)` (two rows), while
`np.hstack((a, b))` gives `(6,)` (one flat array). `vstack` always behaves as if it appended rows,
even when it has to invent the row axis first.

> 🧠 **Design notes**
> - **`concatenate` never broadcasts.** It copies all inputs into one new buffer, and every axis
>   except the chosen one must match exactly — otherwise you get
>   `ValueError: all the input arrays must have same number of dimensions, but the array at index 0
>   has 2 dimension(s) and the array at index 1 has 1 dimension(s)`.
> - **Dtypes are promoted**, exactly as in §5:
>   `np.concatenate([np.arange(3), np.arange(3, dtype=np.float32)]).dtype` is `float64`.
> - **`stack` vs `concatenate` has an easy mnemonic**: `concatenate` glues along an axis the arrays
>   *already have*; `stack` puts the arrays *side by side on a new axis*. When you find yourself
>   writing `a[None]` then `concatenate`, check whether `np.stack` says it more directly — and
>   conversely, when you want to interleave, `stack` + `reshape` is the trick (§ advanced below).
> - **Appending in a loop is O(n²)**: each `np.append` copies everything so far. Collect arrays in a
>   Python list and call `np.concatenate` once. (The same rule applies to `np.vstack` in a loop.)

> 🏋️ **Advanced problem**
> (a) From three `(2, 4)` arrays, build a `(2, 3, 4)` array — state which function you reach for.
> (b) Interleave two 1-D arrays: `[0, 1, 2]` and `[10, 11, 12]` → `[0, 10, 1, 11, 2, 12]`.
> (c) Take a grayscale batch of shape `(N, H, W)` and turn it into `(N, H, W, 3)` without changing
> any pixel values.
> (d) Why is `result = np.zeros(...); for x in data: result = np.append(result, x)` a bug?

<details><summary>Solution</summary>

```python
X1, X2, X3 = (np.ones((2, 4)) * k for k in (1, 2, 3))
np.stack([X1, X2, X3], axis=1).shape          # (a) (2, 3, 4) — a NEW axis in the middle

a = np.array([0, 1, 2]); b = np.array([10, 11, 12])
np.stack((a, b), axis=1).reshape(-1)          # (b) [[0,10],[1,11],[2,12]] -> [0,10,1,11,2,12]

g = np.zeros((8, 32, 32))                     # (c)
np.repeat(g[..., None], 3, axis=-1).shape     # (8, 32, 32, 3)
np.concatenate([g[..., None]] * 3, axis=-1).shape   # same thing

# (d) each append allocates a fresh array and copies everything accumulated so far: total work is
#     O(n^2) for n elements, plus a re-allocation per step. Gather into a list, concatenate once.
```

Part (a) is `stack`, not `concatenate`, because the inputs have only two axes and the result needs
three. Part (c) uses `[..., None]` from §10 to make a length-1 channel axis and then replicates it —
`np.repeat` (repeat elements) and `np.tile` (repeat whole blocks) are the two replication tools.
</details>

---

## 18. Broadcasting

*notebook §8*

> 🐍 **Python native**
> ```python
> table = [[0] * 3 for _ in range(4)]
> for i in range(4):
>     for j in range(3):
>         table[i][j] = i * 10 + (j + 1)
> ```
> The desired output is
> `[[1, 2, 3], [11, 12, 13], [21, 22, 23], [31, 32, 33]]` — a column of multiples of 10 plus a row of
> `1, 2, 3`. Writing it as a double loop hides the structure.

> 🤔 **Surely there's a simpler way?** — The formula `i * 10 + (j + 1)` already *is* an outer sum: a
> column vector plus a row vector. The loops are just how Python has to spell "every combination".

> ✅ **NumPy**
> ```python
> a = np.arange(4) * 10      # (4,)      row vector of 0, 10, 20, 30
> b = np.arange(1, 4)        # (3,)      column vector of 1, 2, 3
> a[:, None] + b             # (4, 1) + (3,) -> (4, 1) + (1, 3) -> (4, 3) -> the table
> # array([[ 1,  2,  3],
> #        [11, 12, 13],
> #        [21, 22, 23],
> #        [31, 32, 33]])
> ```

**The rules** (compare shapes right-to-left, one position at a time):

1. sizes are **equal**, or
2. one of them is **1**, or
3. one of them is **missing** (a shorter shape is padded with 1s on the left).

If every position satisfies one of those, the result takes the largest size at each position and the
length-1 axes are *virtually* stretched. Otherwise you get a `ValueError`.

```
(4, 1)                (3,)  ->  (1, 3)              (2, 3)  vs  (4, 1)
(4, 3)  + (4, 3)?  OK                    (2, 3)  vs  (4, 1)  -> 3 vs 1 OK, 2 vs 4  ERROR
```

Applying the rules to the notebook's quiz with `a` of shape `(4, 3)`:

| candidate | shape | broadcastable with `(4, 3)`? | why |
|---|---|---|---|
| `b = np.arange(1, 4)` | `(3,)` | ✅ | padded to `(1, 3)`: 1 vs 4 OK, 3 vs 3 OK |
| `c = np.arange(1, 5)` | `(4,)` | ❌ | padded to `(1, 4)`: last position 4 vs 3 fails |
| `d = np.ones((4, 1))` | `(4, 1)` | ✅ | 1 stretches to 3 |
| `e = np.ones((1, 3))` | `(1, 3)` | ✅ | 1 stretches to 4 |
| `f = np.ones((2, 3))` | `(2, 3)` | ❌ | first position 2 vs 4 — and neither is 1 |

> 🧠 **Design notes**
> - **Broadcasting does not allocate the stretched data.** The size-1 axis is given a stride of 0, so
>   `(4, 3) + (1, 3)` walks the same row three times. This is why it is both memory-free and fast —
>   and why `np.broadcast_to(x, shape)` returns a **read-only view** rather than a copy.
> - **`None` decides which way the stretching goes**, which is why §10 is a prerequisite. `a[:, None]`
>   gives `(4, 1)` (a column) and `a[None, :]` gives `(1, 4)` (a row); they broadcast against
>   different things.
> - **The comparison is right-aligned**, so "extra" dimensions always go on the left: `(3,)` against
>   `(4, 3)` is really `(1, 3)`. This is why `(4, 1) + (3,)` works out as a *column plus row* rather
>   than an error, and why `(4,) + (4, 3)` fails (it becomes `(1, 4)` vs `(4, 3)`).
> - **`keepdims=True` is the other half of broadcasting**: it keeps a reduction's axis in place as a
>   length-1 axis precisely so it can be broadcast back (§16).
> - Tools when in doubt: `np.broadcast_shapes(s1, s2)` (will this work, and what comes out?) and
>   `np.broadcast_arrays(x, y)` (what are the stretched shapes?).

> 🏋️ **Advanced problem**
> (a) The notebook's exercise: from `a = [0, 1, 2]` and `b = [1, 1, 1, 1, 1]`, produce the `(5, 3)`
> array of five copies of `[0, 1, 2]`.
> (b) Z-score the columns of a `(5, 3)` array in one expression.
> (c) One-hot encode `labels = np.array([0, 3, 1])` for 4 classes with a broadcast comparison.
> (d) Build the `(n, n)` pairwise difference matrix `x_i - x_j` for a `(n, d)` array `X`, then the
> squared-distance matrix.
> (e) Suppose `p` has shape `(4, 1)` and `q` has shape `(5,)`. What is `(p + q).shape`, and how do you
> instead get a `(4, 5)` matrix? What does `(q + p)` give?

<details><summary>Solution</summary>

```python
a = np.arange(3)               # (3,)
b = np.ones(5, dtype=int)      # (5,)
b[:, None] * a + 0             # (5, 1) * (3,) -> (5, 3)      (a)
np.broadcast_to(a, (5, 3))     # cheap read-only way to see the same thing

X = np.random.default_rng(0).normal(size=(5, 3))               # (b)
(X - X.mean(axis=0)) / X.std(axis=0)                           # (5,3) - (3,) -> fine

labels = np.array([0, 3, 1])                                   # (c)
(labels[:, None] == np.arange(4)).astype(int)                  # (3,1) vs (4,) -> (3,4)

X = np.random.default_rng(1).normal(size=(5, 2))               # (d)
X[:, None, :] - X[None, :, :]                                  # (5,1,2) - (1,5,2) -> (5,5,2)
((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)                 # (5, 5)

# (e) right-align: (4,1) and (5,) -> (4,1) vs (1,5) -> (4,5)
p = np.ones((4, 1))
q = np.arange(5)
(p + q).shape           # (4, 5)   the two length-1 axes stretch in DIFFERENT positions
(q + p).shape           # (4, 5)   order of operands does not change the broadcast shape
```

Note how (e) is the *friendly* case: `1` in different positions is exactly what makes an outer
combination, while a `1` next to a mismatching non-1 (like `(2, 3)` against `(4, 3)`) is the error
case. The picture to carry around: right-align the shapes, and each column must be "equal, or 1, or
missing".
</details>

---

# Part IV — Mechanics and tools

## 19. Copies and views in practice

*notebook §9*

> 🐍 **Python native**
> ```python
> import copy
> a = [1, [2, 3]]
> b = a            # alias: b[1][0] = 99 also changes a
> c = a[:]         # shallow copy: c[1][0] = 99 STILL changes a (inner list shared)
> d = copy.deepcopy(a)   # independent
> ```

> 🤔 **Surely there's a simpler way?** — The shallow/deep distinction is already awkward; NumPy adds a
> third tier ("view") that looks like a copy at the syntax level and behaves like an alias at the
> data level. You need a way to *ask* which one you got.

> ✅ **NumPy — ask the array directly**
> ```python
> a = np.arange(6); v = a[1:4]; c = a[[1, 2, 3]]

> np.shares_memory(a, v)      # True    <- the reliable question to ask
> np.shares_memory(a, c)      # False
> v.base is not None          # True    <- v does not own its data
> a.flags.owndata             # True    <- a does own its data
> a.flags.c_contiguous        # True    (the notebook writes `.flags.contiguous`, which does not exist)
> ```

| Operation | View or copy? | Why |
|---|---|---|
| `a[1:4]`, `a[:]`, `a[::2]`, `a[::-1]` | **view** | offset + strides only |
| `a[None]`, `a[:, None]`, `a[...]` | **view** | shape change only |
| `a.T`, `a.transpose(...)`, `a.swapaxes(...)` | **view** | strides permuted |
| `a.reshape(...)` | view **if** the data is contiguous | otherwise a copy |
| `a.ravel()` | view if possible | `a.flatten()` always copies |
| `a.copy()`, `a.astype(...)`, `a[[0, 2]]`, `a[a > 0]` | **copy** | new data must be materialised |
| `a.view(dtype=...)` | **view** | same bytes, reinterpreted |

Why non-contiguous `reshape` copies: `a = np.arange(6).reshape(2, 3).T` has strides `(8, 24)`, i.e.
elements of a row are 24 bytes apart. Asking for shape `(6,)` requires a stride of 8, which cannot be
expressed with the existing layout, so NumPy must copy. `np.shares_memory` confirms it, and
`a.flags.c_contiguous` is the flag that predicts it.

The `view(dtype=...)` oddity from the notebook, explained as byte reinterpretation:

```python
o = np.array([1, 2, 3, 4])     # int64 on Linux/macOS: 8 bytes per element
v = o.view(np.int32)           # 8 int32s: each element split into (low 32 bits | high 32 bits)
v                              # array([1, 0, 2, 0, 3, 0, 4, 0], dtype=int32)
v[0] = 99                      # writes only the low half of element 0
o                              # array([99, 2, 3, 4])   <- the observable effect
```

Viewed as `np.float32` instead, the same bytes decode as IEEE-754 subnormals
(`array([1.e-45, 0.e+00, 3.e-45, ...])`), and writing `99.0` through that view puts the *bit pattern*
of the float into the integer: `o` becomes `array([1120272384, 2, 3, 4])`. The notebook's inline
comment shows `[1120272384 0 3 0]` for the same experiment — that is what you get when the default
integer dtype is 32-bit (Windows and Colab), where each element holds a single float32. **The
principle is the same in both cases and the numbers are the point**: `view(dtype=...)` does not
convert anything, it re-reads the same bytes with a different ruler. Never use it to "convert units".

> 🧠 **Design notes**
> - **Why views at all?** Because slicing is the most common operation in array code, and copying on
>   every `a[1:]` would be catastrophic for memory and time. The cost is the aliasing you can see
>   above.
> - **Why is `a[:]` not a copy, when `l[:]` is?** Because the two types optimise different things:
>   for a list, `l[:]` is the idiomatic copy; for an array, `a[:]` is the idiomatic *whole-array
>   view*. If you want the list behaviour in NumPy, write `a.copy()` and be explicit.
> - **The prediction rule is §13's law**: basic indexing → view; advanced indexing or a
>   dtype-affecting operation → copy. `reshape`/`ravel` are the only "it depends" cases, decided by
>   contiguity.
> - **Gotcha:** `a.flags.contiguous` does not exist (the notebook's §9.5 example); the attributes are
>   `c_contiguous` and `f_contiguous`, and `a.flags['C_CONTIGUOUS']` also works.

> 🏋️ **Advanced problem**
> (a) Prove with `np.shares_memory` that `a[[1, 2]]`, `a[a > 2]`, `a.flatten()` are copies while
> `a[1:3]`, `a.T` and `a[:, None]` are views.
> (b) Modify `a.T` and show the change appears in `a`; then explain why `a.T[0, 0] = 5` is legal even
> though `a.T` is a "temporary".
> (c) Standardise the three columns of a `(4, 3)` array **in place** without creating a second array
> of the same size.
> (d) Does `np.arange(6)[::-1].reshape(2, 3)` share memory with the original? Predict, then check.

<details><summary>Solution</summary>

```python
a = np.arange(6)

np.shares_memory(a, a[[1, 2]])   # False  copy
np.shares_memory(a, a[a > 2])    # False  copy
np.shares_memory(a, a.flatten()) # False  copy
np.shares_memory(a, a[1:3])      # True   view
np.shares_memory(a, a.T)         # True   view (1-D: transpose is a no-op view)
np.shares_memory(a, a[:, None])  # True   view

# (b)
m = np.arange(9).reshape(3, 3)
m.T[0, 0] = 99                   # legal: m.T is a view, so this writes into m
m[0, 0]                          # 99
m[0, 0] = 0

# (c) in-place standardisation: compute into temporary (small) 1-D stats, write into the buffer
X = np.arange(12, dtype=float).reshape(4, 3)
mu = X.mean(axis=0)              # (3,)  small
sd = X.std(axis=0)
X -= mu                          # in-place, m[:, None] broadcasting not even needed
X /= sd
X.mean(axis=0).round(12), X.std(axis=0)     # ~0 and ~1

# (d)
r = np.arange(6)[::-1]
np.shares_memory(r, r.reshape(2, 3))   # True!  a negative-stride view CAN be reshaped
```

Part (d) is a nice refinement of the "non-contiguous means copy" rule: a **uniform** negative stride
is still expressible in the new shape's strides, so no copy is needed, whereas the transposed
`(2, 3)` layout (strides `(8, 24)`) is not. The reliable check is always
`np.shares_memory(a, b)` — never intuition.
</details>

---

## 20. Random numbers

*notebook §10*

> 🐍 **Python native**
> ```python
> import random
>
> random.seed(42)
> [random.random() for _ in range(6)]            # six uniforms, in a loop
> [[random.gauss(5, 2) for _ in range(3)] for _ in range(2)]   # a 2x3 normal sample
> random.randint(0, 9)                           # one integer
> random.sample(range(10), 3)                    # 3 distinct, no loop needed
> random.choices(['a', 'b'], weights=[1, 3], k=5)  # with replacement, weighted
> ```

> 🤔 **Surely there's a simpler way?** — The distribution calls are fine, but every multi-dimensional
> sample needs a nested comprehension, and mixing distributions means mixing `random.*` calls with
> different argument conventions (`random.gauss(mu, sigma)` vs `random.normalvariate`). Statistics
> over samples (`np.mean`, `np.std`) then fall back to a manual loop.

> ✅ **NumPy** (`from numpy import random`)
> ```python
> from numpy import random
> random.seed(42)
> 
> random.rand(2, 3)                       # uniforms in [0, 1), shape given as separate args
> random.random_sample((2, 3))            # the same, shape given as a tuple
> random.uniform(-1, 1, size=5)           # uniform on [-1, 1)
> random.randint(0, 10, size=5)           # ints in [0, 10)
> random.randint(5, 16, size=(2, 2))      # ints in [5, 16)
>
> random.normal(loc=5, scale=2, size=5)   # Normal(mean 5, sd 2)
> random.standard_normal((2, 3))         # N(0, 1)
> random.randn(2, 2)                      # N(0, 1), shape as separate args
>
> random.binomial(n=10, p=0.5, size=5)    # successes in 10 fair coin flips
> random.poisson(lam=3, size=(2, 3))
> random.geometric(p=0.3, size=5)
> random.choice(['apple', 'banana', 'cherry'], size=10, replace=True, p=[.1, .3, .6])
> ```

**The modern API** (recommended since NumPy 1.17, and what new code should use):

```python
rng = np.random.default_rng(42)     # a Generator: an object, not global state
rng.normal(size=5)
rng.integers(0, 10, size=5)
rng.choice(10, size=3, replace=False)
```

> 🧠 **Design notes**
> - **`random.seed(42)` sets *global* state; `default_rng(42)` gives you an object.** Global state
>   means two libraries using randomness can interfere with each other's reproducibility; the
>   Generator API fixes that, and is the documented recommendation. The notebook's `random.seed`
>   still works, and `np.random.seed` is the older equivalent.
> - **The shape argument is inconsistent across functions** — `rand(2, 3)` and `randn(2, 2)` take
>   dimensions as separate positional arguments, while `uniform(..., size=5)`,
>   `normal(..., size=5)` and `randint(..., size=5)` take a tuple. It is a historical wart; remember
>   that `size=(2, 3)` and `rand(2, 3)` produce the same shapes, and that `size=5` and `size=(5,)` are
>   the same thing.
> - **`random.choice` generalises both `random.choice` and `random.choices`** from the stdlib via its
>   `replace=` and `p=` arguments, and it works on any 1-D array-like (including strings).
>   `p` must sum to 1 and have the same length as the population.
> - **Sampling *without* replacement** is `replace=False` (and `p` is then still allowed, but it is
>   slower). For "shuffle", use `rng.permutation(n)` as an index or `rng.permutation(arr)` directly.
> - **Reproducibility is a property of the seed, not of the call.** Any change in the *order* of calls
>   changes every subsequent number, so pin the seed once at the top and never re-seed mid-script.

> 🏋️ **Advanced problem**
> (a) Draw 5 distinct values from `0..99` without replacement, in one call, with the modern API.
> (b) Roll a loaded die (probabilities `[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]`) 4×6 times, as a `(4, 6)`
> array of faces `1..6`.
> (c) Shuffle the rows of an array without copying the rows one at a time.
> (d) Bootstrap: estimate the standard error of the mean of a `(n,)` sample with 1000 resamples, in
> three lines.
> (e) Generate 4000 samples from a bivariate normal with correlation 0.8 using `standard_normal` and
> a Cholesky factor — why 0.8 and not 0.64?

<details><summary>Solution</summary>

```python
rng = np.random.default_rng(0)

rng.choice(100, size=5, replace=False)                       # (a)

p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]                           # (b)
faces = rng.choice(np.arange(1, 7), size=(4, 6), p=p)
faces.shape                                                  # (4, 6)

X = np.arange(12).reshape(4, 3)                              # (c)
X[rng.permutation(X.shape[0])]                               # permutation as an index (a copy)

x = rng.normal(size=50)                                      # (d)
idx = rng.integers(0, x.size, size=(1000, x.size))           # 1000 bootstrap samples
resampled_means = x[idx].mean(axis=1)
resampled_means.std(ddof=1)                                  # ~ standard error of the mean

cov = np.array([[1.0, 0.8], [0.8, 1.0]])                     # (e)
L = np.linalg.cholesky(cov)                                  # lower triangular, L @ L.T == cov
Z = rng.standard_normal((4000, 2)) @ L.T
np.corrcoef(Z.T)[0, 1]                                       # ≈ 0.8
```

Part (d) is the shape trick worth remembering: `rng.integers(0, n, size=(B, n))` draws **B whole rows
of indices at once**, so one fancy-index expression `x[idx]` produces all 1000 bootstrap samples as a
`(1000, n)` array, and `mean(axis=1)` reduces each one. Part (e): the *covariance matrix* is the input
(with 0.8 in the off-diagonal), and the Cholesky factor is what turns independent standard normals
into correlated ones — much faster than a per-sample Python loop.
</details>

---

# Appendices

## Appendix A — Index expression cheat sheet

`a` is any `ndarray`; "ndim-1" means the axis is removed from the result.

| Expression | Plain-Python reading | Meaning | Result | View/copy |
|---|---|---|---|---|
| `a[2]` | `a[2]` | element 2 on axis 0 | scalar (`ndim-1`) | view* |
| `a[-1]` | `a[-1]` | last element on axis 0 | scalar (`ndim-1`) | view* |
| `a[1:4]` | `a[1:4]` | positions 1, 2, 3 (stop excluded) | sub-array, same `ndim` | view |
| `a[:]` | `a[:]` (**beware**: copies a list) | everything on axis 0 | same shape | **view** |
| `a[::2]`, `a[4::2]` | same | every 2nd, optionally with a start | same `ndim` | view |
| `a[::-1]` | same | axis 0 reversed | same shape | view (negative stride) |
| `a[8:2:-1]` | same | backwards from 8, stop excluded | `(6,)` | view |
| `a[5:100]` | — | slices are **clamped** | `(5,)` for `arange(10)` | view |
| `a[0, 1]` | `a[0][1]` | axis 0 = 0 **and** axis 1 = 1 | scalar | view* |
| `a[:, 1]` | `[row[1] for row in a]` | whole column 1 | `(n,)` — axis 1 **dropped** | view |
| `a[:, 1:2]` | — | whole column 1 | `(n, 1)` — axis **kept** | view |
| `a[0]` | `a[0]` | first row | `(m,)` | view |
| `a[1, ...]`, `a[1]` | `a[1]` | sub-array at 1, all remaining axes | `ndim-1` | view |
| `a[..., -1]` | — | last position of the **last** axis, any `ndim` | `ndim-1` | view |
| `a[..., ::-1]` | — | reverse the last axis | same shape | view |
| `a[None]`, `a[:, None]`, `a[..., None]` | `[a]`, `[[x] for x in a]` | **insert** a new length-1 axis | `ndim+1` | view |
| `a[[0, 2, 3, 9]]` | `[a[i] for i in picks]` | gather those positions, in that order | shape of the index | **copy** |
| `a[[2, 2, 0]]` | — | repeats duplicate | shape of the index | copy |
| `a[[0, 1], [1, 2]]` | `[a[i][j] ...]` | the **pairs** (0,1) and (1,2) | broadcast of the index arrays | copy |
| `a[np.ix_([0, -1], [0, -1])]` | — | the **4 corners** (outer combination) | `(2, 2)` | copy |
| `a[[0, -1]][:, [0, -1]]` | — | same block, two index operations | `(2, 2)` | copy |
| `a[a > 4]` | `[x for x in flat if x > 4]` | mask (full shape) | **1-D**, flattened | copy |
| `a[a[:, 0] > 4]` | — | mask along axis 0 | rows selected, `ndim` kept | copy |
| `a[(a > 2) & (a < 6)]` | — | combined masks (parenthesise!) | 1-D | copy |
| `a[np.logical_or(a < 2, a > 6)]` | — | function form of `\|` | 1-D | copy |
| `a[mask] = -1` | — | **scatter** through a mask | writes in place | — |
| `a[[0, -1], :] = 0` | — | scatter into rows | writes in place | — |
| `a[1:, 1:] = b` | — | block assignment (shapes must match) | writes in place | — |
| `a[...] = 0`, `a[:] = 0` | — | fill everything | writes in place | — |
| `np.s_[1:, ::2]` | — | the index expression as an object | reusable index | — |
| `np.r_[0:3, 10]` | `[0, 1, 2] + [10]` | concatenate index ranges/values | `(4,)` | copy |
| `np.c_[x, y]` | `zip` | 1-D arrays as columns | `(n, 2)` | copy |
| `np.take(a, idx, axis=1)` | — | function form of `a[:, idx]` | `a.shape[:1] + np.shape(idx)` | copy |
| `np.take_along_axis(a, idx, axis=1)` | — | per-row gathers (paired rule, no `arange` needed) | `idx.shape` | copy |

\* A "scalar" obtained from an array is a NumPy scalar (`np.int64`), not a Python `int`; it shares the
buffer in the sense that it is a 0-d read of the same data, but writing to it does nothing to `a`.
Assigning through the index (`a[2] = 5`) does.

## Appendix B — Shape rules and the errors they produce

| Index kind | Effect on that axis |
|---|---|
| integer `i` | axis is **removed**; result loses one dimension |
| slice `i:j:k` | axis is **kept**, with the computed length |
| `...` | expands to `:` for every axis not written (at most one `...`) |
| `None` / `np.newaxis` | **new** axis of length 1 is inserted |
| integer array | axis is consumed; the result shape is the **broadcast shape of the index arrays** |
| boolean array, full shape | every `True` element is selected → result is **1-D** |
| boolean array, length `n` on axis 0 | selects **rows**; other axes kept |

Real error messages (NumPy 2.x) — recognise them, they tell you exactly what you did:

```
IndexError: index 10 is out of bounds for axis 0 with size 10
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (3,)
IndexError: boolean index did not match indexed array along axis 0; size of axis is 3 but size of
            corresponding boolean axis is 4
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or
            boolean arrays are valid indices
ValueError: operands could not be broadcast together with shapes (4,3) (2,3)
ValueError: The truth value of an array with more than one element is ambiguous.
            Use a.any() or a.all()
ValueError: could not broadcast input array from shape (1,4) into shape (2,3)
ValueError: cannot reshape array of size 9 into shape (4,3)
TypeError: list indices must be integers or slices, not tuple
```

Read them as diagnostics: "**too many indices**" means you wrote more comma-separated addresses than
the array has axes; "**could not broadcast**" on an assignment means the *right-hand side* does not
fit the target; "**boolean index did not match**" means your mask was built from a different array (or
with `keepdims` missing).

## Appendix C — How to read any index expression in three steps

Take `a[1:, :, ::-1, None]`:

1. **Split on commas.** Four entries → four per-axis address objects: `1:` , `:` , `::-1` , `None`.
   `a[2]` has one entry, so axes 1..n-1 default to `:`.
2. **Classify each entry.** Integer = pick and drop; slice = pick a range and keep; `None` = insert;
   array/list = gather (advanced, copies); bool array = mask (advanced, copies).
3. **Build the shape.** Start from `a.shape`; drop int axes, keep slice axes with their lengths,
   insert a 1 wherever `None` appears, replace consumed axes with the index array's shape, and if any
   entry is advanced, the whole thing is a copy.

The same three steps answer "is this an lvalue?" — every expression is, *as long as you write it as
one `[...]`* rather than chaining (`a[[0,1], 0] = 0` works; `a[[0,1]][:, 0] = 0` silently writes into a
temporary).

## Appendix D — Gotcha list

1. `a * b` is element-wise, `a @ b` is the matrix product. For lists, `a + b` concatenates; for
   arrays it adds.
2. `a[1:2]` keeps the axis (`(1, n)`), `a[1]` drops it (`(n,)`). Use the slice form when broadcasting.
3. `a[:]` **copies** a list and **views** an array. Use `a.copy()` when you mean "independent".
4. Slices are clamped (`a[5:100]` is fine); integer indices are not (`a[100]` raises).
5. `a[1, 2]` is one operation; `a[1][2]` is two. `a[[1]][2]` loses writes (the first step copies).
6. Boolean masks with `and` / `or` / `not` do not work — use `&`, `|`, `~`, with parentheses around
   comparisons.
7. A full-shape mask flattens the result to 1-D. Use `np.where` or a 1-D mask if you need the shape.
8. `np.where(cond, x, y)` evaluates both branches (no short-circuiting); `np.maximum(a, 0)` is
   clearer and faster than `np.where(a > 0, a, 0)`.
9. `axis=` is the axis that **disappears**; add `keepdims=True` before combining the result with the
   original array.
10. `arr.shape` is a tuple, so `arr.shape[0]`, `arr.shape[-1]`, `arr.shape[:-1] + (1,)` are ordinary
    Python.
11. `reshape` may return a view or a copy; `flatten()` always copies, `ravel()` copies only if it must,
    and `np.shares_memory(a, b)` is the only reliable check.
12. `a[[0, 0]] += 1` adds once, not twice — use `np.add.at` or `np.bincount` when indices repeat.
13. `np.append` and `np.vstack` inside a loop are O(n²); build a list and call `np.concatenate` once.
14. Never compare floats with `==` on arrays; use `np.allclose` / `np.isclose`.
15. `view(dtype=...)` reinterprets raw bytes; it does not convert values.
16. Integer dtypes are platform dependent (32-bit on Windows, 64-bit on Linux/macOS) — printf-style
    assumptions about exact values are not portable.

---

## Where this leaves you

The notebook covers much more than the syntax: vectorisation, array creation, `dtype`, indexing,
assignment, arithmetic, reductions, stacking, broadcasting, views/copies and the random module. The
recurring idea behind all of it is the one in §0: **an array is a shape plus strides over a buffer**,
and every piece of syntax above is either "compute an address" (indexing, `reshape`, `transpose`,
`None`) or "do one C loop over the buffer" (arithmetic, reductions, broadcasting).

If you want to practise, redo the notebook's in-cell exercises from memory (§6.2, §6.3, §6.4 and §8
are the meaty ones), then re-derive each advanced problem here with the solutions hidden. The two
habits that pay off fastest: write down the **shape** before every serious expression, and ask
"is this a view or a copy?" whenever you are about to write through it.

*Built from `lab1_review.ipynb` (COMP 2211 Lab 1: NumPy). All snippets verified against NumPy 2.5.x.*
