# 2D Cell-Centered Unstructured FVM Poisson (MPI + METIS)

This is a minimal but structured 2D unstructured finite volume (cell-centered) Poisson solver framework:

- 2D Fluent ASCII `.cas` input only (no binary, no 3D)
- Gauss-Seidel (SOR) linear solver with MPI halo exchange
- METIS graph partitioning for cell-based domain decomposition
- Tecplot ASCII output (cell centers) with `u_num`, `u_exact`, `error`

## Build (Linux)

```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

Requires: MPI, METIS, CMake >= 3.15, C++17 compiler.

## Run

Single rank:
```bash
./build/poisson2d --case cases/demo
```

MPI:
```bash
mpirun -np 4 ./build/poisson2d --case cases/demo
```

## Configuration (INI)

Config file is required. Program accepts:

- `./poisson2d --case <caseDir>` -> uses `<caseDir>/config.ini`

Example `config.ini`:
```
[case]
mesh_file = cav_1_20.cas
output_dir = output

[solver]
maxIter = 2000
tol = 1e-10
printInterval = 20
checkInterval = 1
omega = 1.0
```

`checkInterval` controls how often global residual is evaluated (`MPI_Allreduce`).
For large MPI runs, setting `checkInterval=5~20` can reduce synchronization overhead.

## Recommended Build Types for Performance

- Debug (`-O0 -g`): for debugging only
- Release (`-O3`): preferred for production timing/scaling
- RelWithDebInfo: profiling with symbols and optimization

All other parameters must be in config.

## Input Mesh (.cas) Requirements

- 2D Fluent ASCII `.cas` only
- Tri/quad cells (faces are 2-node edges in 2D)
- Parser supports common Fluent ASCII sections:
  - `(10 ...)` nodes
  - `(13 ...)` faces
  - `(12 ...)` cells (optional; used for count if present)
- Cell IDs and node IDs are expected 1-based as in Fluent
- Boundary faces are detected by neighbor cell ID = 0

### How to Export Fluent ASCII .cas (conceptual)

In Fluent/ANSYS:
1. Ensure the mesh is 2D.
2. Use the Case export/save option and choose ASCII `.cas`.
3. Do not export binary format.

## Discretization Summary

Poisson equation:
```
-(d2u/dx2 + d2u/dy2) = f
u_exact = sin(pi*x) * sin(pi*y)
f = 2*pi^2 * sin(pi*x) * sin(pi*y)
```

Cell-centered FVM with orthogonal diffusion flux:
- Internal face: `(uN - uP) * (|Sf| / dPN)`
- Dirichlet boundary: `(uB - uP) * (|Sf| / dPB)`

`dPN` is the projection of cell-center distance onto face normal.  
`uB` is the exact solution at the face center.

## Outputs

Output directory (from config):

- `residual.csv` (iter,residual)
- `solution.dat` (Tecplot ASCII; cell centers)

The log prints L2 and Linf errors:
```
L2 = sqrt( sum(Vc*(u_num-u_exact)^2) / sum(Vc*(u_exact)^2) )
Linf = max |u_num-u_exact|
```

## Limitations

- 2D only
- Fluent ASCII `.cas` only
- No non-orthogonal correction (first-order orthogonal approximation)
- Gauss-Seidel only (SOR supported via `omega`)

## TODO / Extensions

- Non-orthogonal correction and improved gradient reconstruction
- Faster solvers (CG, AMG, PETSc)
- More boundary condition types (Neumann, mixed)
- Binary `.cas` and 3D support
