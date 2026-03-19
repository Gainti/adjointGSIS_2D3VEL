#pragma once

// #include "dvmSolver.h"
// #include "meshDeform.h"
// #include "boundary_sensitivity.h"
// #include "mpi.h"

// double computeBoundaryObjective(
//     dvmSolver& solver,
//     const BoundaryFunctional& obj)
// {
//     const auto& mesh = solver.mesh;
//     double J_local = 0.0;

//     for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
//         const auto& f = mesh.faces[facei];
//         if (f.bc_type != BCType::wall &&
//             f.bc_type != BCType::pressure_far_field) {
//             continue;
//         }

//         double A = f.Sf.mag();
//         vector nf = f.Sf / A;
//         int ghost = f.neigh;

//         for (int vi = 0; vi < solver.Nv; ++vi) {
//             double cn = nf.x * solver.Vx[vi] + nf.y * solver.Vy[vi];
//             int idx_g = solver.index_vdf(ghost, vi);

//             double h = solver.vdf[idx_g];
//             double m = (cn > 0.0) ? obj.mPlus(solver, facei, vi)
//                                   : obj.mMinus(solver, facei, vi);

//             J_local += A * cn * m * h * solver.feq[vi] * solver.weight[vi];
//         }
//     }

//     double J_global = 0.0;
//     MPI_Allreduce(&J_local, &J_global, 1, MPI_DOUBLE, MPI_SUM, solver.comm);
//     return J_global;
// }

// double projectGradientToYScalingDirection(
//     const Mesh& mesh,
//     const std::vector<NodeGrad>& nodeGrad,
//     MPI_Comm comm)
// {
//     std::vector<char> isBoundary(mesh.points.size(), 0);
//     for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
//         const auto& f = mesh.faces[facei];
//         isBoundary[f.n1] = 1;
//         isBoundary[f.n2] = 1;
//     }

//     double local = 0.0;
//     for (size_t pid = 0; pid < mesh.points.size(); ++pid) {
//         if (!isBoundary[pid]) continue;
//         local += nodeGrad[pid].dy * mesh.points[pid].y;
//     }

//     double global = 0.0;
//     MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
//     return global;
// }
// void buildYScalingDisplacement(
//     const Mesh& globalMesh,
//     double scale,
//     std::vector<BoundaryNodeDisplacement>& bdisp)
// {
//     std::vector<int> boundaryPts;
//     collectAllBoundaryPoints(globalMesh, boundaryPts);

//     bdisp.clear();
//     bdisp.reserve(boundaryPts.size());

//     for (int pid : boundaryPts) {
//         const auto& point = globalMesh.points[pid];
//         vector d(0.0, (scale - 1.0) * point.y, 0.0);
//         bdisp.push_back({pid, d});
//     }
// }
// struct SensitivityResult
// {
//     double J = 0.0;
//     double directionalDeriv = 0.0;
//     std::vector<FaceGeomGrad> faceGrad;
//     std::vector<NodeGrad> nodeGrad;
// };

// SensitivityResult solveAdjointAndAssembleSensitivity(
//     Mesh& localMesh,
//     const SolverConfig& scfg,
//     MPI_Comm comm)
// {
//     SensitivityResult out;

//     dvmSolver dvm(localMesh, scfg, comm);

//     for (int iter = 1; iter < scfg.max_iter; ++iter) {
//         dvm.step(iter);
//         if (dvm.res_ux < scfg.tol &&
//             dvm.res_uy < scfg.tol &&
//             dvm.res_rho < scfg.tol) {
//             break;
//         }
//     }

//     dvm.initialAdj();
//     for (int iter = 1; iter < scfg.max_iter; ++iter) {
//         dvm.stepAdj(iter);
//         if (dvm.res_aux < scfg.tol &&
//             dvm.res_auy < scfg.tol &&
//             dvm.res_arho < scfg.tol) {
//             break;
//         }
//     }

//     XForceFunctional obj;

//     BoundarySensitivityAssembler::assembleFaceGradients(dvm, obj, out.faceGrad);
//     BoundarySensitivityAssembler::accumulateNodeGradients(localMesh, out.faceGrad, out.nodeGrad);

//     out.J = computeBoundaryObjective(dvm, obj);
//     out.directionalDeriv = projectGradientToYScalingDirection(localMesh, out.nodeGrad, comm);

//     return out;
// }

// double solvePrimalAndEvaluateObjective(
//     Mesh& localMesh,
//     const SolverConfig& scfg,
//     MPI_Comm comm)
// {

//     dvmSolver dvm(localMesh, scfg, comm);

//     for (int iter = 1; iter < scfg.max_iter; ++iter) {
//         dvm.step(iter);
//         if (dvm.res_ux < scfg.tol &&
//             dvm.res_uy < scfg.tol &&
//             dvm.res_rho < scfg.tol) {
//             break;
//         }
//     }

//     XForceFunctional obj;
//     return computeBoundaryObjective(dvm, obj);
// }

// void verifySensitivityByFiniteDifference(
//     const Mesh& globalMesh0,
//     const Mesh& localMesh0,
//     const SolverConfig& scfg,
//     MPI_Comm comm,
//     double eps)
// {
//     int rank = 0;
//     MPI_Comm_rank(comm, &rank);

//     // --------------------------------------------------
//     // 1) 基准网格：伴随导数
//     // --------------------------------------------------
//     Mesh localBase = localMesh0;
//     auto baseResult = solveAdjointAndAssembleSensitivity(localBase, scfg, comm);

//     // --------------------------------------------------
//     // 2) scale = 1 + eps
//     // --------------------------------------------------
//     Mesh globalPlus = globalMesh0;
//     Mesh localPlus  = localMesh0;

//     {
//         std::vector<BoundaryNodeDisplacement> bdisp;
//         buildYScalingDisplacement(globalPlus, 1.0 + eps, bdisp);
//         deformMeshSpring(globalPlus, localPlus, bdisp, comm, 500, 1e-10, 1.8);
//     }

//     double Jplus = solvePrimalAndEvaluateObjective(localPlus, scfg, comm);

//     // --------------------------------------------------
//     // 3) scale = 1 - eps
//     // --------------------------------------------------
//     Mesh globalMinus = globalMesh0;
//     Mesh localMinus  = localMesh0;

//     {
//         std::vector<BoundaryNodeDisplacement> bdisp;
//         buildYScalingDisplacement(globalMinus, 1.0 - eps, bdisp);
//         deformMeshSpring(globalMinus, localMinus, bdisp, comm, 500, 1e-10, 1.8);
//     }

//     double Jminus = solvePrimalAndEvaluateObjective(localMinus, scfg, comm);

//     // --------------------------------------------------
//     // 4) 中心差分
//     // --------------------------------------------------
//     double fd = (Jplus - Jminus) / (2.0 * eps);
//     double adj = baseResult.directionalDeriv;

//     double absErr = std::abs(fd - adj);
//     double relErr = absErr / std::max(std::abs(fd), 1e-14);

//     if (rank == 0) {
//         printf("=====================================================\n");
//         printf("Finite-difference sensitivity verification\n");
//         printf("eps                 = %.12e\n", eps);
//         printf("J(base)             = %.12e\n", baseResult.J);
//         printf("J(1+eps)            = %.12e\n", Jplus);
//         printf("J(1-eps)            = %.12e\n", Jminus);
//         printf("FD directional deriv= %.12e\n", fd);
//         printf("Adj directional deriv= %.12e\n", adj);
//         printf("Abs error           = %.12e\n", absErr);
//         printf("Rel error           = %.12e\n", relErr);
//         printf("=====================================================\n");
//     }
// }