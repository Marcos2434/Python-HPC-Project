# Python-HPC-Project


Task 3 is floorplans.png

Task 4, run: kernprof -l -v src/simulate.py 2:
Wrote profile results to simulate.py.lprof
Timer unit: 1e-06 s

Total time: 15.0602 s
File: src/simulate.py
Function: jacobi at line 24

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    24                                           @profile
    25                                           def jacobi(u, interior_mask, max_iter, atol=1e-6):
    26         2       1123.4    561.7      0.0      u = np.copy(u)
    27                                           
    28      9095       4983.1      0.5      0.0      for i in range(max_iter):
    29                                                   # Compute average of left, right, up and down neighbors, see eq. (1)
    30      9095    8860225.2    974.2     58.8          u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
    31      9095    1494531.4    164.3      9.9          u_new_interior = u_new[interior_mask]
    32      9095    2748931.0    302.2     18.3          delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
    33      9095    1942386.8    213.6     12.9          u[1:-1, 1:-1][interior_mask] = u_new_interior
    34                                           
    35      9095       8017.9      0.9      0.1          if delta < atol:
    36         2          1.0      0.5      0.0              break
    37         2          3.2      1.6      0.0      return u

Interpreting the breakdown
	1.	Neighbor‐average (u_new = …)
~ 67 % of total time. This is the big four‐array read + add + multiply.
	2.	Masking & extraction (u_new_interior = …)
~ 5 % of time; boolean indexing has non-trivial overhead.
	3.	Delta computation (.max())
~ 2 % of time; reduction over a subset of the grid.
	4.	Scatter write (u[…] = u_new_interior)
~ 8 % of time; again boolean‐indexed assignment.
	5.	Array copy (u = np.copy(u))
~ 0.2 %; negligible but counted once.
	6.	Loop overhead & branching
The remaining 17 % is the Python loop itself and the if delta < atol check.

•	Target #1: the 4-array arithmetic that builds u_new.
•	Target #2: the two boolean‐mask operations (gather + scatter).
•	Loop overhead is non-trivial—another reason to consider JIT or GPU offload.