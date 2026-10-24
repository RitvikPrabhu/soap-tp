# Distributed symmetric eigendecomposition, step by step

This is a guide to the algorithm in [`eigh.cpp`](eigh.cpp). It computes all
eigenvalues and eigenvectors of a **real symmetric** matrix on AMD GPUs.
There is one execution path, in native FP32 or FP64, without fallback solvers.

## 1. What are we trying to compute?

Given a symmetric matrix \(A=A^T\), find an orthogonal matrix \(Q\) and a diagonal
matrix \(\Lambda\) such that

\[
A Q = Q\Lambda, \qquad Q^TQ=I.
\]

Column \(q_j\) of \(Q\) is an eigenvector: multiplying it by \(A\) scales it by
\(\lambda_j\). A negative eigenvalue reverses its direction; zero sends it to
zero. Orthogonal means that the columns have unit length and are mutually
perpendicular.

We do not try to solve this dense problem directly. First we change coordinates
to make the matrix **tridiagonal**: only the main diagonal and the two adjacent
diagonals may be nonzero. That simpler matrix has the same eigenvalues.
After solving its eigenproblem, we change the eigenvectors back.

```diagram
┌────────────────────────────┐
│ Dense symmetric A          │
└─────────────┬──────────────┘
              │ Householder coordinate changes
              ▼
┌────────────────────────────┐
│ Tridiagonal T              │
└─────────────┬──────────────┘
              │ ELPA tridiagonal eigensolve
              ▼
┌────────────────────────────┐
│ Eigenvalues and vectors Z  │
└─────────────┬──────────────┘
              │ Undo coordinate changes: Q = U Z
              ▼
┌────────────────────────────┐
│ Original-matrix vectors Q  │
└────────────────────────────┘
```

The custom code owns the dense-to-tridiagonal reduction and the final change of
coordinates. **ELPA supplies the distributed tridiagonal eigensolver**, not a
replacement dense solve or a fallback.

## 2. Divide the matrix among the GPUs

Code: `Grid`.

Arrange MPI ranks into a \(p_r\)-by-\(p_c\) process grid, with one rank per GPU.
Cut the matrix into square blocks of width \(b\), and distribute these blocks
cyclically along both grid dimensions. For zero-based global entry \((i,j)\),
its owner is

\[
\left(\lfloor i/b\rfloor\bmod p_r,\;
      \lfloor j/b\rfloor\bmod p_c\right).
\]

For example, on a 2-by-4 grid the first block row belongs to process row 0,
the next to process row 1, and the next returns to process row 0. Block columns
similarly cycle through process columns 0, 1, 2, 3.

Each rank stores its own entries in column-major order. It stores both triangles
of its local matrix, not just the lower triangle. Separate row and column
communicators let ranks exchange vectors and sum partial products without
collecting the dense matrix on one GPU.

The tridiagonal stage uses a separate \((p_r p_c)\)-by-1 grid. Both grids must
give every rank nonempty ownership; for eight ranks this requires
\(n>7b\).

## 3. Scale the matrix without changing its eigenvectors

Code: `eigh`, `matrix_max`, `matrix_scale`.

Each GPU finds its largest absolute local entry. An MPI maximum gives the
largest entry over the whole matrix. We choose a power of two \(s\) and work on

\[
\widehat A=A/s.
\]

For a nonzero finite matrix, the largest absolute scaled entry is between
one-half and one. A zero matrix is left unchanged. Power-of-two scaling avoids
an extra rounding step when the result remains representable, and keeps the
FP32 tridiagonal problem away from unnecessarily small or large magnitudes.

Scaling changes eigenvalues, not eigenvectors. At the end we multiply the
computed eigenvalues by \(s\) to recover the original scale.

## 4. Use one reflection to eliminate most of a column

Code: `write_reflector`, `local_statistics`, `reflector_scalar`, `form_vectors`.

Consider column \(k\). Allow its diagonal and first subdiagonal positions to
remain nonzero; eliminate everything below that first subdiagonal:

```text
Before reflection             After reflection

  *  *  *  *                    *  *  0  0
  *  *  *  *                    *  *  *  *
  *  *  *  *                    0  *  *  *
  *  *  *  *                    0  *  *  *

Shown for k = 0. Symmetry eliminates matching row entries too.
The stars are not necessarily unchanged values.
```

Write the entries below the diagonal as \(x=[\alpha;\,y]\). A Householder
reflection has the form

\[
H=I-\tau vv^T.
\]

When \(y\ne0\), choose

\[
\beta=-\operatorname{copysign}\left(\sqrt{\alpha^2+\|y\|_2^2},\alpha\right),
\quad v=\begin{bmatrix}1\\y/(\alpha-\beta)\end{bmatrix},
\quad \tau=(\beta-\alpha)/\beta.
\]

Then \(Hx=[\beta;0;\ldots;0]\). The sign choice avoids subtracting two nearly
equal values in \(\alpha-\beta\). If \(y=0\), nothing needs eliminating:
the code sets \(\tau=0\), making \(H=I\).

For the full matrix, pad \(v\) with zeros in positions 0 through \(k\).
The reflection therefore acts only on the remaining coordinates. Applying
\(H\) on **both sides**, \(A\leftarrow HAH\), preserves symmetry and eigenvalues.

Norms and reflector scalars are formed in FP64, even with FP32 matrix state.
Local norms use a maximum scale before squaring; the distributed calculation
combines these scaled norm contributions rather than squaring raw large values.

## 5. Apply the reflection without constructing H

Code: `reduce`, `vector_dot`, `correct_vectors`, `local_rank2`.

An explicit dense \(H\) would waste storage and work. Instead, on the active
trailing matrix compute

\[
w_0=\tau A v,\qquad
w=w_0-\frac{\tau}{2}(v^T w_0)v,
\qquad A\leftarrow A-vw^T-wv^T.
\]

Expanding \((I-\tau vv^T)A(I-\tau vv^T)\) gives exactly this rank-two update
in exact arithmetic. Here “rank two” means two vector outer products, not two
MPI processes.

The distributed work for one column is:

1. Broadcast the active column across its process row.
2. Combine its norm contributions across process rows to construct the reflector.
3. Arrange the reflector in the row and column layouts required by local products.
4. Use rocBLAS GEMV for each GPU's contribution to \(Av\); sum contributions
   across process columns with RCCL.
5. Sum the scalar dot product across process rows and form \(w\).
6. Use a HIP kernel to update each GPU's own trailing entries.

The column layout needs portions owned by different process rows, so the code
uses padded RCCL AllGather buffers. These replicate vectors, not the dense matrix.
With only one process row, the local vector already contains the required data.

Repeat for successive columns. This implementation uses **one reflector at a
time**, not a delayed multi-column reduction panel.

## 6. Keep the reflectors instead of constructing U

Code: `reduce` and `form_vectors`.

After all columns have been processed, mathematically

\[
T=U^T\widehat A U,\qquad U=H_0 H_1\cdots H_{n-2}.
\]

We do not store dense \(U\). We store each reflector vector in the corresponding
distributed matrix column, and keep its scalar \(\tau_k\). The tridiagonal
diagonal and off-diagonal are saved separately as arrays `d` and `e`.

Thus the overwritten matrix buffer contains **reflector storage**, not a literal
dense copy of \(T\). The tridiagonal problem needs only the two length-\(n\)
arrays. The reflector scalars are also replicated; reflector vectors remain
stored in the distributed matrix and are communicated as needed.

## 7. Solve the smaller-structure eigenproblem

Code: `tridiagonal`.

Pass `d` and `e` to ELPA's public distributed tridiagonal component to obtain

\[
TZ=Z\Lambda.
\]

ELPA supplies its distributed tridiagonal eigensolution machinery, including
divide-and-conquer. This code does not implement that machinery itself.
The public API accepts host arrays and manages its GPU work internally.

For FP64 state, call `elpa_solve_tridiagonal_d`. For FP32 state, cast `d` and `e`
to FP32 and call `elpa_solve_tridiagonal_f`. Copying FP32 eigenvalues into the
driver's FP64 output container afterward does **not** recover FP64 accuracy.

The resulting eigenvector matrix \(Z\) is distributed on the all-process-rows
grid. ScaLAPACK `PDGEMR2D` or `PSGEMR2D` redistributes it to the original process
grid, retaining its precision. There is no dense gather or second eigensolve.

## 8. Transform the eigenvectors back

Code: `backtransform` and `compact_t`.

We have eigenvectors of \(T\), but want eigenvectors of \(A\). Since
\(T=U^T\widehat A U\), the answer is

\[
Q=UZ=H_0 H_1\cdots H_{n-2}Z.
\]

Order matters: start from \(Z\), apply the rightmost reflector first, and work
backward. Do not reverse the mathematical product itself.

To use matrix-matrix multiplication instead of many matrix-vector operations,
group up to 256 consecutive reflectors. A group's product can be represented as

\[
H_{\mathrm{start}}\cdots H_{\mathrm{end}-1}=I-VCV^T,
\]

where \(V\) contains the group's reflector columns and \(C\) is a small upper
triangular matrix. This is the **compact-WY representation**; `compact_t` calls
this small matrix `t`, distinct from the tridiagonal matrix \(T\).

For each group, in reverse group order:

1. Broadcast the stored reflector columns into a row-distributed \(V\).
2. Compute \(V^TV\) using local GEMMs and a reduction across process rows.
3. Construct \(C\) from those inner products and the stored reflector scalars.
   This small calculation uses CPU FP64 BLAS.
4. Apply the group using three GEMMs:
   \(Y=V^TQ\), then \(W=CY\), then \(Q\leftarrow Q-VW\).
   Reduce the partial \(Y\) matrices across process rows before using them.

After the last group, the columns of \(Q\) correspond to the ascending
eigenvalues. In `eigh`, the eigenvalue scale from step 3 is restored before
calling this backtransform; no eigenvector rescaling is needed.

## 9. Check that the result solves the original problem

Code: `run` and `product`.

The standalone driver generates deterministic test matrices. After its final
solve repetition, it checks the result against the original, unscaled matrix:

\[
r=\frac{\|AQ-Q\Lambda\|_F}{\|A\|_F},\qquad
o=\|Q^TQ-I\|_F.
\]

The Frobenius norm is the square root of the sum of squared entries. The first
check asks whether each output column is an eigenvector with its claimed
eigenvalue. The second asks whether the eigenvectors form an orthonormal basis.

Validation uses distributed FP64 matrix products, outside solver timing. For
the zero matrix, whose relative residual is undefined, it checks the absolute
residual. Thresholds are \(r\le10^{-10}\) for FP64 and \(r\le10^{-4}\) for FP32;
the orthogonality threshold is the corresponding value times \(\sqrt n\).

For \(n\le512\), a separate CPU LAPACK solve also checks eigenvalues. Only this
small validation reference constructs a full matrix on rank zero. Larger
problems have no independent eigenvalue reference in this driver.

A failed check exits nonzero. It does not retry, promote the solve, or switch
algorithms. Small residual and orthogonality errors do not guarantee accurate
individual eigenvectors in tightly clustered eigenspaces, or agreement between
two full SOAP optimizer trajectories.

## Reading the source in order

Start with `eigh`: scaling → `reduce` → `tridiagonal` → `backtransform`.
Then read `reduce` alongside its HIP kernels. Finally read `run` for matrix
generation, timing, and numerical validation. `Grid` owns distribution and
communication; `Device` owns GPU allocations.

The required stack is HIP/gfx90a, rocBLAS, RCCL, MPI, GPU-enabled ELPA, and
BLACS/ScaLAPACK/CPU BLAS/LAPACK. The driver accepts `--n`, `--block`, `--pr`,
`--pc`, `--precision`, `--matrix`, and `--repeat`. There is no Python/SOAP binding,
mixed-precision mode, alternate eigensolver, or host-communication fallback.
