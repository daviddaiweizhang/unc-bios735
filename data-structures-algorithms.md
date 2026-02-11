# Basics of Data Structures and Algorithms

## Introduction

In the previous lecture on [computational complexity](computational-complexity.md), we learned how to analyze the time and space costs of algorithms using Big-O notation. We also saw that choosing the right approach (brute force vs. sorting vs. hashing) can change an algorithm from impractical to instant. The key insight was that **algorithm choice matters more than hardware**.

This lecture builds on that foundation. We'll study the core data structures and algorithm design techniques that underpin efficient scientific computing. The goal is not to memorize implementations, but to develop intuition for *when and why* to reach for each tool.

We will cover:

* Data structures beyond Python's built-ins: linked lists, stacks, queues, trees, and graphs
* Recursion and backtracking
* Sorting algorithms and how they compare
* Algorithm design paradigms: divide and conquer, greedy algorithms, and dynamic programming
* Graph algorithms: BFS, DFS, and Dijkstra's shortest path

These topics come up constantly in scientific computing. Graph algorithms appear in network analysis and phylogenetics. Dynamic programming is the foundation of sequence alignment in bioinformatics. Trees power database indexing and spatial search. Even if you never implement these from scratch, understanding them helps you choose the right library and anticipate performance bottlenecks.


## Data Structures

We already know Python's built-in data structures: lists, dictionaries, sets, and tuples. These cover many use cases, but some problems call for more specialized structures.

### Stacks and Queues

A **stack** follows Last-In-First-Out (LIFO) order: the most recently added item is the first to be removed, like a stack of plates. A **queue** follows First-In-First-Out (FIFO) order: items are processed in the order they arrived, like a line at a store.

Python lists can serve as stacks (use `append` and `pop`), but for queues the `collections.deque` is more efficient because it supports $O(1)$ operations on both ends.

```python
# Stack using a Python list
stack = []
stack.append("task A")
stack.append("task B")
stack.append("task C")
print(stack.pop())  # "task C" (last in, first out)
print(stack.pop())  # "task B"
print(stack)        # ["task A"]
```

```python
# Queue using collections.deque
from collections import deque

queue = deque()
queue.append("patient 1")
queue.append("patient 2")
queue.append("patient 3")
print(queue.popleft())  # "patient 1" (first in, first out)
print(queue.popleft())  # "patient 2"
print(queue)            # deque(["patient 3"])
```

Stacks are used for function call management (the "call stack"), undo/redo operations, and expression parsing. Queues are used for job scheduling, breadth-first search (which we'll see later), and managing task pipelines.

### Linked Lists

A **linked list** is a sequence of nodes where each node stores a value and a reference (pointer) to the next node. Unlike a Python list (which is backed by a contiguous array), linked list elements are scattered in memory and connected by pointers.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements))


ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.display()  # 10 -> 20 -> 30
```

The key trade-off between linked lists and arrays:

| Operation | Array (Python list) | Linked List |
|---|---|---|
| Access by index | $O(1)$ | $O(n)$ |
| Insert/delete at beginning | $O(n)$ | $O(1)$ |
| Insert/delete at end | $O(1)$ amortized | $O(n)$ without tail pointer |
| Search | $O(n)$ | $O(n)$ |

Linked lists are useful when you need frequent insertions and deletions at arbitrary positions and don't need random access. In practice, Python's `collections.deque` (a doubly-linked list internally) is the go-to when you need efficient operations on both ends.

### Trees

A **tree** is a hierarchical data structure where each node has zero or more children. The topmost node is the **root**, and nodes with no children are **leaves**.

A **binary tree** is a tree where each node has at most two children (left and right). A **binary search tree (BST)** adds an ordering property: for every node, all values in the left subtree are smaller, and all values in the right subtree are larger. This enables $O(\log n)$ search, insertion, and deletion (when the tree is balanced).

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def insert_bst(root, value):
    """Insert a value into a binary search tree."""
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insert_bst(root.left, value)
    else:
        root.right = insert_bst(root.right, value)
    return root


def search_bst(root, target):
    """Search for a value in a binary search tree."""
    if root is None:
        return False
    if target == root.value:
        return True
    if target < root.value:
        return search_bst(root.left, target)
    return search_bst(root.right, target)


def inorder_traversal(root):
    """Visit nodes in sorted order: left, root, right."""
    if root is None:
        return []
    return inorder_traversal(root.left) + [root.value] + inorder_traversal(root.right)


# Build a BST
root = None
for val in [50, 30, 70, 20, 40, 60, 80]:
    root = insert_bst(root, val)

print(inorder_traversal(root))  # [20, 30, 40, 50, 60, 70, 80]
print(search_bst(root, 40))    # True
print(search_bst(root, 45))    # False
```

Trees appear throughout scientific computing:

* **Decision trees** in machine learning split data based on feature thresholds
* **KD-trees** partition spatial data for efficient nearest-neighbor queries (we mentioned this in the complexity lecture)
* **Heap data structures** (a special kind of binary tree) power priority queues, which are used in Dijkstra's algorithm and event-driven simulations
* **File systems** are tree structures
* **Phylogenetic trees** represent evolutionary relationships in biology


### Graphs

A **graph** consists of **vertices** (nodes) and **edges** (connections between nodes). Graphs are one of the most versatile data structures in computing. An edge can be **directed** (one-way, like a web link) or **undirected** (two-way, like a friendship). Edges can also have **weights** representing costs, distances, or strengths.

The two standard ways to represent a graph are:

**Adjacency list:** Each vertex stores a list of its neighbors. Space-efficient for sparse graphs (few edges relative to vertices).

**Adjacency matrix:** A 2D matrix where entry $(i, j)$ indicates whether an edge exists between vertices $i$ and $j$. Enables $O(1)$ edge lookup but uses $O(V^2)$ space.

```python
# Adjacency list representation (using a dictionary)
graph_list = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

# Check if an edge exists
print("B" in graph_list["A"])  # True
print("F" in graph_list["A"])  # False
```

```python
import numpy as np

# Adjacency matrix representation
# Vertices: A=0, B=1, C=2, D=3, E=4, F=5
adj_matrix = np.array([
    [0, 1, 1, 0, 0, 0],  # A
    [1, 0, 0, 1, 1, 0],  # B
    [1, 0, 0, 0, 0, 1],  # C
    [0, 1, 0, 0, 0, 0],  # D
    [0, 1, 0, 0, 0, 1],  # E
    [0, 0, 1, 0, 1, 0],  # F
])

# Check if edge exists between A (0) and B (1)
print(adj_matrix[0, 1])  # 1 (edge exists)
```

| Representation | Space | Edge lookup | Iterate neighbors |
|---|---|---|---|
| Adjacency list | $O(V + E)$ | $O(\text{degree})$ | $O(\text{degree})$ |
| Adjacency matrix | $O(V^2)$ | $O(1)$ | $O(V)$ |

Real-world graph applications include:

* **Social networks:** people are vertices, friendships are edges
* **Protein interaction networks:** proteins are vertices, interactions are edges
* **Citation networks:** papers are vertices, citations are directed edges
* **Road networks:** intersections are vertices, roads are weighted edges

#### Question

You are analyzing a protein interaction network with 20,000 proteins where each protein interacts with an average of 10 others. Would you choose an adjacency list or an adjacency matrix? How much memory does each representation use (roughly, in bytes)?

#### Answer

Choose an **adjacency list**. With 20,000 proteins and ~10 interactions each, there are about 100,000 edges. An adjacency list stores these as pairs, requiring roughly $O(V + E) = O(20{,}000 + 100{,}000) \approx 120{,}000$ entries, around 1 MB.

An adjacency matrix would require $20{,}000 \times 20{,}000 = 4 \times 10^8$ entries. As a byte matrix, that's 400 MB, or as float64, 3.2 GB. Most of these entries would be zero since the graph is sparse (100,000 edges out of 200 million possible), wasting memory.


## Recursion and Backtracking

Recursion is when a function calls itself to solve smaller instances of the same problem. Every recursive function needs a **base case** (when to stop) and a **recursive case** (how to reduce the problem).

We saw Fibonacci in the complexity lecture. Here is another classic example: computing the factorial of a number.

```python
def factorial(n):
    """Compute n! recursively."""
    if n <= 1:       # base case
        return 1
    return n * factorial(n - 1)  # recursive case

print(factorial(5))  # 120 = 5 * 4 * 3 * 2 * 1
```

Python has a default recursion limit of 1000 (check with `sys.getrecursionlimit()`). For deep recursion, you may need to increase it or convert to an iterative approach.

### Backtracking

**Backtracking** is a technique that builds solutions incrementally, abandoning ("backtracking" from) a partial solution as soon as it determines the solution cannot be completed successfully. It's essentially a depth-first exploration of the solution space with pruning.

A classic example is generating all permutations of a list:

```python
def permutations(elements):
    """Generate all permutations of elements using backtracking."""
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])  # found a complete permutation
            return
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i + 1 :])
            current.pop()  # undo the choice (backtrack)

    backtrack([], elements)
    return result


perms = permutations([1, 2, 3])
print(f"Number of permutations: {len(perms)}")
for p in perms:
    print(p)
```

The backtracking pattern has three steps: (1) make a choice, (2) recursively explore, (3) undo the choice. This pattern appears in constraint satisfaction problems, puzzle solving, and combinatorial optimization.


## Sorting Algorithms

Sorting is one of the most fundamental operations in computing. Python's built-in `sorted()` and `list.sort()` use **Timsort**, a hybrid algorithm with $O(n \log n)$ worst-case performance. Understanding different sorting algorithms helps you appreciate why $O(n \log n)$ is the theoretical lower bound for comparison-based sorting and how different strategies achieve it.

### Insertion Sort

Insertion sort builds the sorted list one element at a time by inserting each new element into its correct position among the already-sorted elements. It's like sorting a hand of playing cards.

```python
def insertion_sort(arr):
    """Sort array in place using insertion sort. O(n^2)."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


data = [64, 25, 12, 22, 11]
print(f"Before: {data}")
insertion_sort(data)
print(f"After:  {data}")
```

Insertion sort is $O(n^2)$ in the worst case but $O(n)$ when the input is already nearly sorted. This makes it a good choice for small arrays or as a final pass in hybrid algorithms (Timsort uses insertion sort for small subarrays).

### Merge Sort

Merge sort is a **divide and conquer** algorithm: it splits the array in half, recursively sorts each half, then merges the two sorted halves. It guarantees $O(n \log n)$ time in all cases.

```python
def merge_sort(arr):
    """Sort array using merge sort. O(n log n)."""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    """Merge two sorted arrays into one sorted array."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


data = [38, 27, 43, 3, 9, 82, 10]
print(f"Before: {data}")
sorted_data = merge_sort(data)
print(f"After:  {sorted_data}")
```

The key insight: merging two sorted arrays of total size $n$ takes $O(n)$ time. The array is split $\log n$ times, and at each level the total merge work is $O(n)$, giving $O(n \log n)$ overall.

### Quicksort

Quicksort is another divide and conquer algorithm. It picks a **pivot** element, partitions the array so that elements less than the pivot come before it and elements greater come after it, then recursively sorts each partition. (We briefly saw this in the first Python lecture.)

```python
def quicksort(arr):
    """Sort array using quicksort. O(n log n) average, O(n^2) worst."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


data = [38, 27, 43, 3, 9, 82, 10]
print(f"Before: {data}")
sorted_data = quicksort(data)
print(f"After:  {sorted_data}")
```

Quicksort is $O(n \log n)$ on average but $O(n^2)$ in the worst case (when the pivot is always the smallest or largest element). Despite this, quicksort is often faster than merge sort in practice because it sorts in-place (our simplified version above creates new lists, but the in-place variant avoids this overhead) and has good cache behavior.

### Comparing Sorting Algorithms

```python
import time
import numpy as np


def time_sort(sort_func, data, repetitions=5):
    """Time a sorting function over multiple repetitions."""
    total = 0
    for _ in range(repetitions):
        arr = data.copy()
        start = time.time()
        sort_func(arr)
        total += time.time() - start
    return total / repetitions


for n in [1000, 5000, 10000]:
    data = list(np.random.randint(0, n * 10, n))
    t_insert = time_sort(insertion_sort, data)
    t_merge = time_sort(merge_sort, data)
    t_quick = time_sort(quicksort, data)
    t_builtin = time_sort(sorted, data)
    print(
        f"n={n:>6d}: insertion={t_insert:.4f}s, merge={t_merge:.4f}s, "
        f"quick={t_quick:.4f}s, built-in={t_builtin:.6f}s"
    )
```

| Algorithm | Best | Average | Worst | Space | Stable? |
|---|---|---|---|---|---|
| Insertion sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | Yes |
| Merge sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes |
| Quicksort | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ | No |
| Timsort (Python) | $O(n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes |

A "stable" sort preserves the relative order of equal elements. This matters when sorting by multiple keys (e.g., sort by name, then by date).

Here is a [visualization of different sorting algorithms](https://www.youtube.com/watch?v=kPRA0W1kECg).

#### Question

You have a list of 10 million patient records that are *almost sorted* (only a few hundred records are out of place due to data entry corrections). Which sorting algorithm would perform best here, and why?

#### Answer

**insertion sort** would perform best. Timsort is designed to exploit existing order in the data and runs in $O(n)$ on nearly-sorted input. Insertion sort is also $O(n)$ on nearly-sorted data since each element only needs to move a few positions. Merge sort and quicksort would both be $O(n \log n)$ regardless of the existing order. For 10 million records, use the built-in `sorted()` since Timsort handles this case optimally and is implemented in C.


## Algorithm Design Paradigms

When faced with a new problem, algorithm designers often reach for one of several standard strategies. We'll cover three of the most important: divide and conquer, greedy algorithms, and dynamic programming.

### Divide and Conquer

We already saw divide and conquer with merge sort and quicksort. The pattern is:

1. **Divide** the problem into smaller subproblems
2. **Conquer** each subproblem recursively
3. **Combine** the results

Another classic application is **binary search**, which finds a target value in a sorted array by repeatedly halving the search range:

```python
def binary_search(arr, target):
    """Find target in sorted array. Returns index or -1. O(log n)."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(binary_search(data, 23))  # 5
print(binary_search(data, 50))  # -1
```

Binary search is $O(\log n)$ because each comparison eliminates half the remaining elements. It requires sorted input, but if you perform many searches on the same data, the one-time $O(n \log n)$ sorting cost is worth it.

Divide and conquer appears widely in scientific computing:

* **Fast Fourier Transform (FFT)** reduces $O(n^2)$ convolution to $O(n \log n)$
* **Strassen's algorithm** multiplies matrices in $O(n^{2.807})$ instead of $O(n^3)$
* **Closest pair of points** algorithms run in $O(n \log n)$ instead of $O(n^2)$

### Greedy Algorithms

A **greedy algorithm** makes the locally optimal choice at each step, hoping that local choices lead to a global optimum. Greedy algorithms are typically simple and fast, but they don't always produce the best solution.

A classic example is the **activity selection problem**: given a set of activities with start and end times, select the maximum number of non-overlapping activities.

```python
def activity_selection(activities):
    """Select maximum non-overlapping activities.

    Args:
        activities: List of (start, end) tuples.

    Returns:
        List of selected activities.
    """
    # Greedy strategy: always pick the activity that ends earliest
    sorted_acts = sorted(activities, key=lambda x: x[1])
    selected = [sorted_acts[0]]
    last_end = sorted_acts[0][1]

    for start, end in sorted_acts[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected


activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11)]
selected = activity_selection(activities)
print(f"Selected {len(selected)} activities: {selected}")
```

This greedy approach is provably optimal for activity selection. The key insight is that choosing the earliest-ending activity leaves the most room for future activities.

However, greedy algorithms can fail. Consider the **coin change problem**: make change for a given amount using the fewest coins.

```python
def greedy_coin_change(amount, denominations):
    """Greedy coin change: always pick the largest coin possible."""
    denominations = sorted(denominations, reverse=True)
    coins_used = []
    remaining = amount
    for coin in denominations:
        while remaining >= coin:
            coins_used.append(coin)
            remaining -= coin
    return coins_used


# Works for standard US denominations
print(greedy_coin_change(36, [1, 5, 10, 25]))
# [25, 10, 1] = 3 coins (optimal)

# Fails for non-standard denominations
print(greedy_coin_change(6, [1, 3, 4]))
# [4, 1, 1] = 3 coins (greedy), but [3, 3] = 2 coins (optimal!)
```

Greedy algorithms work well when the problem has the **greedy choice property** (a locally optimal choice can be extended to a globally optimal solution) and **optimal substructure** (optimal solutions contain optimal solutions to subproblems).

### Dynamic Programming

**Dynamic programming (DP)** solves problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the results. It's more powerful than greedy algorithms because it considers all possible choices rather than committing to the locally best one.

We already saw the Fibonacci example in the complexity lecture, where memoization reduced $O(2^n)$ to $O(n)$. Here's a more practical example.

**The 0/1 Knapsack Problem:** Given items with weights and values, and a knapsack with a weight capacity, find the maximum value you can carry.

```python
def knapsack(weights, values, capacity):
    """Solve 0/1 knapsack problem using dynamic programming.

    Args:
        weights: List of item weights.
        values: List of item values.
        capacity: Maximum weight capacity.

    Returns:
        Maximum achievable value.
    """
    n = len(weights)
    # dp[i][w] = max value using first i items with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: don't take item i
            dp[i][w] = dp[i - 1][w]
            # Option 2: take item i (if it fits)
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1],
                )

    return dp[n][capacity]


# Example: lab equipment selection with budget constraint
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8
print(f"Maximum value: {knapsack(weights, values, capacity)}")
```

The key DP concepts are:

* **Overlapping subproblems:** The same subproblems appear repeatedly (unlike divide and conquer, where subproblems are independent)
* **Optimal substructure:** The optimal solution contains optimal solutions to subproblems
* **Two approaches:** Top-down (recursion with memoization) or bottom-up (fill a table iteratively)

A scientifically important DP application is **sequence alignment**, widely used in bioinformatics to compare DNA, RNA, or protein sequences:

```python
def edit_distance(s1, s2):
    """Compute the minimum edit distance between two strings.

    The edit distance (Levenshtein distance) counts the minimum number
    of insertions, deletions, and substitutions to transform s1 into s2.
    This is the core of sequence alignment algorithms like Needleman-Wunsch.
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = edit distance between s1[:i] and s2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming to/from empty string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # characters match
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


# Compare DNA sequences
seq1 = "AGCTGAC"
seq2 = "ACGTGCA"
print(f"Edit distance between {seq1} and {seq2}: {edit_distance(seq1, seq2)}")

# Compare words
print(f"Edit distance between 'kitten' and 'sitting': {edit_distance('kitten', 'sitting')}")
```

The time and space complexity of this DP solution is $O(mn)$, where $m$ and $n$ are the lengths of the two sequences. The brute-force approach of trying all possible alignments would be exponential.

#### Question

How would you compare the greedy, divide and conquer, and dynamic programming approaches? For each, name one problem where it's the right choice and one where it would fail or be suboptimal.

#### Answer

**Divide and conquer** splits problems into *independent* subproblems. Right choice: merge sort (subproblems don't overlap). Would fail: Fibonacci (subproblems overlap massively, leading to exponential redundant work without memoization).

**Greedy** makes locally optimal choices without reconsidering. Right choice: activity selection (earliest-end-time strategy is provably optimal). Would fail: 0/1 knapsack (taking the highest value-to-weight ratio item can miss the global optimum).

**Dynamic programming** handles problems with *overlapping* subproblems. Right choice: edit distance / sequence alignment (exponentially many overlapping subproblems). Would fail (in the sense of being overkill): binary search, where subproblems don't overlap and divide and conquer suffices.


## Graph Algorithms

Graphs model relationships, and many computational problems reduce to traversing or finding paths in graphs. We'll cover the three most fundamental graph algorithms.

### Breadth-First Search (BFS)

BFS explores a graph level by level, visiting all neighbors of a vertex before moving to their neighbors. It uses a **queue** and finds the shortest path (by number of edges) from a starting vertex to all reachable vertices.

```python
from collections import deque


def bfs(graph, start):
    """Breadth-first search. Returns vertices in BFS order."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

print("BFS from A:", bfs(graph, "A"))
```

BFS can be extended to find the shortest path (in terms of number of edges):

```python
def bfs_shortest_path(graph, start, goal):
    """Find shortest path from start to goal using BFS."""
    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        vertex, path = queue.popleft()
        if vertex == goal:
            return path
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # no path found


path = bfs_shortest_path(graph, "A", "F")
print(f"Shortest path A -> F: {path}")
```

BFS time complexity is $O(V + E)$ where $V$ is the number of vertices and $E$ is the number of edges. It's used for finding shortest paths in unweighted graphs, computing connected components, and level-order tree traversal.

### Depth-First Search (DFS)

DFS explores as deep as possible along each branch before backtracking. It uses a **stack** (either explicitly or via recursion).

```python
def dfs_recursive(graph, start, visited=None):
    """Depth-first search using recursion."""
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))
    return order


def dfs_iterative(graph, start):
    """Depth-first search using an explicit stack."""
    visited = set()
    stack = [start]
    order = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            # Add neighbors in reverse order so we visit them left-to-right
            for neighbor in reversed(graph[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


print("DFS (recursive) from A:", dfs_recursive(graph, "A"))
print("DFS (iterative) from A:", dfs_iterative(graph, "A"))
```

DFS also runs in $O(V + E)$ time. It's used for cycle detection, topological sorting (ordering tasks with dependencies), and finding connected components. The key difference from BFS is the exploration pattern: BFS goes wide, DFS goes deep.

#### Question

You are mapping a social network and want to find if two people are connected through at most 3 intermediate friends (i.e., within 4 hops). Would you use BFS or DFS? Why?

#### Answer

Use **BFS**. BFS explores the graph level by level, so it naturally finds the shortest path between two nodes. You can stop as soon as you find the target (guaranteeing the shortest path) or once you've explored all nodes up to depth 4. DFS could find a path, but it might follow a very long path first and wouldn't guarantee finding the shortest one without exhaustive search.

![DFS xkcd comic](https://imgs.xkcd.com/comics/dfs.png)


## Summary

* **Data structures** provide different trade-offs. Stacks (LIFO) and queues (FIFO) control processing order. Linked lists enable efficient insertions/deletions. Binary search trees provide $O(\log n)$ lookup when balanced. Graphs model relationships between entities.

* **Recursion** solves problems by reducing them to smaller instances. Backtracking extends recursion by exploring choices and undoing them when they fail.

* **Sorting algorithms** differ in their guarantees. Insertion sort is fast on nearly-sorted data ($O(n)$ best case). Merge sort guarantees $O(n \log n)$. Quicksort is $O(n \log n)$ on average but $O(n^2)$ worst case. Python's built-in Timsort combines the best of both.

* **Algorithm design paradigms** guide how to approach new problems. Divide and conquer splits into independent subproblems (merge sort, binary search, FFT). Greedy algorithms make locally optimal choices (activity selection, Huffman coding). Dynamic programming stores solutions to overlapping subproblems (knapsack, sequence alignment, edit distance).

* **Graph algorithms** solve network problems. BFS finds shortest unweighted paths in $O(V + E)$. DFS explores deeply for cycle detection and topological sorting in $O(V + E)$. Dijkstra's finds shortest weighted paths in $O((V + E) \log V)$.


## Recommended Resources

* [Big-O Cheat Sheet](https://www.bigocheatsheet.com/): complexity reference for common algorithms and data structures
* [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition): the standard reference, covers all topics in depth
* [Visualgo](https://visualgo.net/): interactive visualizations of sorting algorithms, graph traversals, and more
* [Python `collections` module](https://docs.python.org/3/library/collections.html): `deque`, `Counter`, `defaultdict`, and other useful data structures
* [NetworkX](https://networkx.org/): Python library for graph analysis with built-in BFS, DFS, Dijkstra's, and many more algorithms
* [Biological Sequence Analysis (Durbin et al.)](https://www.cambridge.org/core/books/biological-sequence-analysis/921BB77E4E4B3BFC0F8B65B3FCFE641D): dynamic programming for sequence alignment in bioinformatics
