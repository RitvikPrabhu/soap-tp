# Distributed symmetric eigendecomposition, step by step

This is a guide to the algorithm in [`eigh.cpp`](eigh.cpp). It computes all
eigenvalues and eigenvectors of a **real symmetric** matrix on AMD GPUs.
There is one execution path, in native FP32 or FP64, without fallback solvers.

Read sections 1 and 3–8 for the mathematics. Section 2 and the communication
parts of sections 5 and 8 explain how the same operations are distributed.
Section 10 works through a complete small example with actual numbers.

### Notation used throughout

| Symbol | Meaning |
| --- | --- |
| \(A\) | Original real symmetric matrix, size \(n\times n\) |
| \(\widehat A\) | Power-of-two-scaled input |
| \(A^{(k)}\) | Mathematical matrix after \(k\) reflections, before reflector storage overwrites eliminated entries |
| \(H_k\) | Full-size reflection eliminating column \(k\) below its first subdiagonal |
| \(U\) | Product of all the reflections; never explicitly constructed |
| \(T\) | Tridiagonal matrix obtained from the reduction |
| \(Z\) | Eigenvectors of \(T\) |
| \(Q\) | Eigenvectors of the original matrix |
| \(\widehat\Lambda\), \(\Lambda\) | Eigenvalues before and after restoring the input scale |

All mathematical identities below describe exact arithmetic. The precision
notes explain where the implementation rounds these operations.

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

### Why changing coordinates preserves the eigenproblem

An orthogonal matrix satisfies \(U^{-1}=U^T\). Suppose we construct
\(T=U^T\widehat A U\) and solve \(TZ=Z\widehat\Lambda\). Then

\[
\widehat A(UZ)=U(U^T\widehat A U)Z=UTZ=UZ\widehat\Lambda.
\]

Therefore \(Q=UZ\) contains eigenvectors of \(\widehat A\). If
\(\widehat A=A/s\), multiplying both sides by \(s\) gives
\(AQ=Q(s\widehat\Lambda)\). This explains both final operations: change the
vectors back with \(U\), and restore the eigenvalue scale separately.

Orthogonality also survives: if \(Z^TZ=I\), then
\(Q^TQ=Z^TU^TUZ=I\). We choose orthogonal coordinate changes because they
preserve Euclidean lengths and do not themselves introduce an ill-conditioned
change-of-basis matrix.

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

### Global indices versus local indices

For a global row \(i\) owned by a given process row, its local row index is

\[
\ell_r(i)=\left\lfloor\frac{i}{bp_r}\right\rfloor b+(i\bmod b).
\]

The analogous local column index uses \(p_c\). `Grid::li` computes this map.
For example, with block size 2 and two process rows, process row 0 owns global
rows 0, 1, 4, 5, 8, 9, …, stored locally as rows 0, 1, 2, 3, 4, 5, ….
The `rows` and `cols` arrays are the inverse maps: local indices to global ones.

Let \(I_r\) and \(J_c\) be the global row and column sets owned by process row
\(r\) and process column \(c\). Rank \((r,c)\) holds
\(A_{rc}=A[I_r,J_c]\), with local dimensions \(n_r\times n_c\).
Those dimensions can differ between ranks when the final blocks are partial.
The grid is block-cyclic, not a simple contiguous partition into rectangles.

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

More precisely, `frexp` chooses an integer exponent \(e\) such that the maximum
entry magnitude is \(m2^e\), with \(1/2\le m<1\). The GPU scales each entry
using `scalbn(value, -e)`; it does not need to construct \(s=2^e\) explicitly.
The output eigenvalues use `scalbn(value, e)`.

This normalization does **not** improve the matrix's condition number or widen
relative eigengaps. It rescales all eigenvalues together. It is a numerical
range-management step, not a remedy for nearly indistinguishable eigenvectors.

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

### Deriving the reflector rather than memorizing it

For a nonzero tail \(y\), we want a reflection taking \(x\) to \(\beta e_1\),
where \(e_1\) is the first coordinate vector of the active subspace. Reflections
preserve length, so \(|\beta|=\|x\|_2\). Choose its sign as above and define

\[
u=x-\beta e_1,\qquad H=I-\frac{2uu^T}{u^Tu}.
\]

Because \(\beta^2=x^Tx\),

\[
u^Tx=\beta(\beta-\alpha),\qquad
u^Tu=2\beta(\beta-\alpha).
\]

Consequently \(Hx=x-u=\beta e_1\). Also \(H^T=H\), and expanding \(H^2\)
gives \(I\), so \(H\) is both symmetric and orthogonal.

The code normalizes the first entry of \(u\) to one:
\(v=u/(\alpha-\beta)\). Substituting this into the expression for \(H\)
gives \(\tau=2/(v^Tv)=(\beta-\alpha)/\beta\). Storing \(v\) and one scalar
is enough; no dense reflection matrix is needed.

### Why earlier zeros stay zero

At iteration \(k\), \(H_k\) is the identity on coordinates 0 through \(k\).
For any earlier column \(j<k\), all its entries in coordinates \(k+1\) onward
are already zero. Reflecting those zero entries cannot make them nonzero.
That is why processing columns in order produces tridiagonal form without
undoing earlier work.

### How the distributed norm is combined

For the tail entries owned by process row \(r\), compute
\(a_r=\max_i|y_i|\) and \(t_r=\sum_i(y_i/a_r)^2\), using zero contribution
when \(a_r=0\). After gathering these pairs, compute

\[
a=\max_r a_r,\qquad
\|y\|_2=a\sqrt{\sum_r t_r(a_r/a)^2}.
\]

When the whole tail is zero, return norm zero without division. `local_statistics`
produces the local pairs, and `reflector_scalar` combines them. The unique
owners of the pivot and diagonal also contribute those scalars. This is a
mathematical zero case, not a fallback algorithm.

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

### Deriving the rank-two update

Let \(B=A^{(k)}[k+1:n,k+1:n]\) be the active trailing block and use the active
portion of \(v\). Expanding the two-sided reflection gives

\[
HBH=B-\tau v(v^TB)-\tau(Bv)v^T
       +\tau^2(v^TBv)vv^T.
\]

Since \(B\) is symmetric, \(v^TB=(Bv)^T\). Define
\(w_0=\tau Bv\) and \(\gamma=v^Tw_0\). With
\(w=w_0-(\tau/2)\gamma v\), we obtain

\[
vw^T+wv^T
=\tau v(Bv)^T+\tau(Bv)v^T-\tau^2(v^TBv)vv^T.
\]

Subtracting that expression from \(B\) gives exactly \(HBH\). The factor
one-half is essential: the two outer products each contribute half of the
last term. Only the trailing block is updated this way; the diagonal and new
subdiagonal of column \(k\) are saved in `d[k]` and `e[k]`.

### Which communicator sums which quantity?

In the next equations, restrict \(I_r\) and \(J_c\) to active indices greater
than \(k\), and interpret \(B\)'s indices as their original global coordinates.
For the distributed product, split \(v\) into entries indexed by \(J_c\).
Each rank forms a partial row result

\[
p_{rc}=B[I_r,J_c]v[J_c],\qquad (Bv)[I_r]=\sum_c p_{rc}.
\]

This sum is **across process columns**, so it uses `g.row`. The resulting
row-layout vector is replicated across process columns. Next,

\[
\gamma=\sum_r v[I_r]^T w_0[I_r]
\]

is a sum **across process rows**, so it uses `g.col`. A world-wide sum here
would incorrectly count the process-column replicas multiple times.

`v` and `w` contain row-layout entries; `vc` and `wc` contain column-layout
entries. For a local entry with global indices \((i,j)\), `local_rank2` performs
the scalar operation \(B_{ij}\leftarrow B_{ij}-v_iw_j-w_iv_j\).

To construct the column layout from row-owned segments, AllGather uses the
common padded length
\(L=\lceil n/(bp_r)\rceil b\). For global entry \(i\), the gathered offset is
\(\operatorname{ownerRow}(i)L+\ell_r(i)\). Padding makes every rank's send
count equal; the global-to-local map ensures padding is never used as a matrix
entry. This is the indexing in `form_vectors` and `correct_vectors`.

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

For instance, after reducing column 0 the mathematical entries at rows 2 onward
of that column are zero. The implementation reuses that space for the nonzero
reflector entries. It also stores the leading one at row 1 and zeros at rows
0 through \(k\) of each stored reflector column. Future reduction steps only
read the still-active trailing block, so this storage does not contaminate
their matrix products.

The mathematical invariant after \(k\) steps is

\[
A^{(k)}=(H_0\cdots H_{k-1})^T\widehat A(H_0\cdots H_{k-1}).
\]

`d[k]` records the diagonal before replacing column \(k\) with a reflector;
`e[k]` records its new first subdiagonal \(\beta\). After the loop, the single
owner of the bottom-right matrix entry supplies the final diagonal. Only the
first \(n-1\) off-diagonal entries represent edges of the tridiagonal matrix.

## 7. Solve the smaller-structure eigenproblem

Code: `tridiagonal`.

Pass `d` and `e` to ELPA's public distributed tridiagonal component to obtain

\[
TZ=Z\widehat\Lambda.
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

### What divide-and-conquer means mathematically

The following explains the underlying idea, not every internal ELPA special
case. Split a tridiagonal matrix at an off-diagonal entry \(b\). If \(b\ne0\),
let \(\rho=|b|\), subtract \(\rho\) from the adjacent diagonal entry on each
side of the split, and call the resulting two blocks \(T_1,T_2\). Then

\[
T=\begin{bmatrix}T_1&0\\0&T_2\end{bmatrix}+\rho uu^T,
\qquad u=\begin{bmatrix}e_{\mathrm{last}}\\
                  \operatorname{sign}(b)e_{\mathrm{first}}\end{bmatrix}.
\]

The outer product restores both subtracted diagonal values and the coupling
\(b\). If \(b=0\), the two blocks are already independent. Solve each smaller
block, writing the combined block-diagonal eigendecomposition as \(VDV^T\).
In those coordinates the remaining problem is

\[
D+\rho zz^T,\qquad z=V^Tu.
\]

For an eigenvalue \(\theta\) away from the diagonal entries of \(D\), its
eigenvector satisfies
\((D-\theta I)y=-\rho z(z^Ty)\). Eliminating \(y\) yields the **secular equation**

\[
1+\rho\sum_i\frac{z_i^2}{d_i-\theta}=0.
\]

Thus a merge combines two solved subproblems through a structured rank-one
update instead of restarting a dense eigensolve. Repeated eigenvalues, zero
couplings, deflation, and stable root finding need numerical treatment; those
belong to ELPA, not to this source file. We reuse that tested component while
keeping the custom dense reduction and backtransform explicit.

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

### Why reflector order cannot be swapped

After two steps, \(T=H_1H_0\widehat A H_0H_1\). Therefore
\(U=H_0H_1\), not \(H_1H_0\). To form \(UZ\), first compute \(H_1Z\), then
apply \(H_0\). Individual reflectors are symmetric, but their product generally
is not, and different reflectors generally do not commute.

### Deriving the small triangular matrix

Suppose an existing group has product \(I-VCV^T\). Append a reflector
\(I-\tau vv^T\) on its right. Expanding the product shows that its new compact
representation uses

\[
V_{\mathrm{new}}=[V\;v],\qquad
C_{\mathrm{new}}=
\begin{bmatrix}
C&-\tau C(V^Tv)\\
0&\tau
\end{bmatrix}.
\]

This is exactly `compact_t`: read the previous-column inner products from the
Gram matrix, multiply them by \(-\tau\), apply the existing upper triangular
matrix with `dtrmv`, and put \(\tau\) on the new diagonal. For two reflectors,
the off-diagonal is \(-\tau_0\tau_1 v_0^Tv_1\); ignoring that term would fail
to represent their product.

For a group width \(h\), each GPU holds \(V_r\) of shape \(n_r\times h\),
replicated across process columns. The small Gram matrix is
\(G=\sum_r V_r^TV_r\), of shape \(h\times h\). For local output columns,

\[
Y_c=\sum_r V_r^TQ_{rc},\qquad
W_c=CY_c,\qquad Q_{rc}\leftarrow Q_{rc}-V_rW_c.
\]

Here \(Y_c,W_c\) have shape \(h\times n_c\). These equations explain both
the local GEMM dimensions and the process-column communicator reductions
(`g.col`) in `backtransform`.

The fixed width 256 groups operations into GPU-friendly GEMMs. It does not
change the mathematical product or mean the reduction also uses width 256.
The reduction still forms reflectors sequentially, one at a time.

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

The driver first generates entries in FP64, then casts them to the solve's
native precision. Its residual and CPU reference use the original FP64 entries,
not their rounded FP32 copies. Therefore an FP32 validation result includes
input-quantization error as well as eigensolver error. The eigenvalue check is
\(\max_i|\lambda_i-\lambda_i^{\mathrm{ref}}|/\max_i|\lambda_i^{\mathrm{ref}}|\),
with the same precision-specific tolerance and an absolute error for a zero
reference spectrum; it is not an individual relative error for each eigenvalue.

A failed check exits nonzero. It does not retry, promote the solve, or switch
algorithms. Small residual and orthogonality errors do not guarantee accurate
individual eigenvectors in tightly clustered eigenspaces, or agreement between
two full SOAP optimizer trajectories.

The checks are deliberately about the eigenproblem, not entrywise equality of
two eigenvector matrices. Eigenvectors may differ by sign, and any orthonormal
basis of a repeated-eigenvalue subspace is valid. Near-repeated eigenvalues can
also make individual vectors sensitive even when the residual is small.

## 10. A complete numerical example

Ignore scaling temporarily and take

\[
A=\begin{bmatrix}4&3&4\\3&2&0\\4&0&2\end{bmatrix}.
\]

**Eliminate the bottom entry of column 0.** The active column is \(x=[3,4]^T\).
Its norm is 5, so \(\alpha=3\), \(\beta=-5\),
\(v=[1,1/2]^T\), and \(\tau=8/5\). The active reflection and its full-size
embedding are

\[
H_{\mathrm{active}}=
\begin{bmatrix}-3/5&-4/5\\-4/5&3/5\end{bmatrix},\qquad
H_0=\begin{bmatrix}1&0&0\\0&-3/5&-4/5\\0&-4/5&3/5\end{bmatrix}.
\]

Direct multiplication gives \(H_{\mathrm{active}}[3,4]^T=[-5,0]^T\).
The trailing block is \(B=2I\). For the rank-two update,
\(w_0=\tau Bv=[3.2,1.6]^T\), \(v^Tw_0=4\), and
\(w=w_0-0.8\cdot4v=0\). So the trailing block stays \(2I\), while the
first column and row change as intended:

\[
T=H_0^TAH_0=
\begin{bmatrix}4&-5&0\\-5&2&0\\0&0&2\end{bmatrix}.
\]

The remaining reflection is the identity because there is no tail to eliminate.
Thus \(U=H_0\). Notice how this example distinguishes **updating the trailing
block** from **recording the new subdiagonal**: a zero rank-two correction does
not mean the entire column was unchanged.

**Solve the tridiagonal problem.** One eigenvalue is 2, with eigenvector
\(z=[0,0,1]^T\). The upper 2-by-2 block has characteristic equation

\[
(4-\lambda)(2-\lambda)-25=\lambda^2-6\lambda-17=0.
\]

Its eigenvalues are \(3-\sqrt{26}\) and \(3+\sqrt{26}\), approximately −2.09902
and 8.09902. For either root \(\lambda\), an eigenvector is proportional to
\([5,4-\lambda,0]^T\); divide by its Euclidean norm to make it unit length.
ELPA handles the general distributed tridiagonal version of this stage.

**Return to the original coordinates.** For the middle eigenvalue 2,

\[
q=Uz=\begin{bmatrix}0\\-4/5\\3/5\end{bmatrix},\qquad
Aq=\begin{bmatrix}0\\-8/5\\6/5\end{bmatrix}=2q.
\]

For either other root, multiply the normalized tridiagonal eigenvector by
\(H_0\) in the same way. Its unnormalized original-coordinate direction is
\([5,-(3/5)(4-\lambda),-(4/5)(4-\lambda)]^T\).
The reflection preserves its norm, so the same normalization works.

**Connect this example to scaling.** The actual code sees maximum entry 4 and
chooses scale 8. It reduces \(A/8\), gets the same reflector, solves for
eigenvalues divided by 8, and multiplies them back by 8. The final eigenvectors
are unchanged. This small example illustrates the algebra; an eight-rank run
requires a larger matrix to satisfy the nonempty-ownership constraint.

## 11. Precision and cost: what this implementation does and does not do

| Quantity or operation | FP32 run | FP64 run |
| --- | --- | --- |
| Dense matrix and reflector-vector storage | FP32 | FP64 |
| rocBLAS GEMV/GEMM and rank-two matrix update | FP32 | FP64 |
| Scaled norm and dot-product scalar accumulation | FP64 | FP64 |
| `d`, `e`, and `tau` containers during reduction | FP64 | FP64 |
| ELPA tridiagonal solve and its eigenvectors | FP32 | FP64 |
| Small compact-WY triangular recurrence | FP64, then cast to FP32 | FP64 |
| Distributed eigenvector output | FP32 | FP64 |
| Driver's eigenvalue container | FP64 container holding FP32-derived values | FP64 |
| Residual/orthogonality validation | FP64 | FP64 |

FP64 scalar work does not undo rounding already incurred in an FP32 GEMV or
matrix update. For example, `vector_dot` scales and stores each \(w_0\) entry
in the native type, then accumulates the dot product in FP64.

For active dimension \(m\), forming \(Bv\) and updating \(B\) each require work
proportional to \(m^2\). Summing over the reduction gives cubic work in \(n\).
Applying the stored reflectors to all \(n\) eigenvectors is also cubic work;
compact-WY makes that work efficient matrix multiplication rather than removing
it. The tridiagonal matrix is cheap to store, but computing all its eigenvectors
still requires a dense distributed output.

Each rank stores its local dense matrices, plus vector and narrow-panel
workspaces. Dense storage is distributed; vectors and small Gram matrices may
be replicated. There are sequential reflector dependencies and collectives
throughout the reduction, so adding GPUs does not imply linear speedup.

The implementation choices are intentionally narrow: orthogonal Householder
steps preserve the symmetric eigenproblem; native arithmetic gives a clear
precision contract; ELPA owns the difficult tridiagonal merge machinery; and
compact-WY exposes the final dense work to rocBLAS. No alternate dense solve,
mixed-precision experiment, or communication fallback is hidden in that path.

## Reading the source in order

Start with `eigh`: scaling → `reduce` → `tridiagonal` → `backtransform`.
Then read `reduce` alongside its HIP kernels. Finally read `run` for matrix
generation, timing, and numerical validation. `Grid` owns distribution and
communication; `Device` owns GPU allocations.

The required stack is HIP/gfx90a, rocBLAS, RCCL, MPI, GPU-enabled ELPA, and
BLACS/ScaLAPACK/CPU BLAS/LAPACK. The driver accepts `--n`, `--block`, `--pr`,
`--pc`, `--precision`, `--matrix`, and `--repeat`. There is no Python/SOAP binding,
mixed-precision mode, alternate eigensolver, or host-communication fallback.
