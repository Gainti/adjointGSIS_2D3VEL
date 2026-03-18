#include "adjoint_validation.h"


#include "computeGeometry.h"

#include <cstdio>
#include <cmath>
#include <algorithm>

/* ============================================================
 * 小工具
 * ============================================================ */

static double dot2(double ax, double ay, double bx, double by)
{
    return ax * bx + ay * by;
}

static double norm2(double x, double y)
{
    return std::sqrt(x * x + y * y);
}

bool isBoundaryFace(const Face& f)
{
    return static_cast<int>(f.bc_type) != static_cast<int>(BCType::internal);
}

double localReferenceLength(const Mesh& mesh, int point_id)
{
    double s = 0.0;
    int cnt = 0;
    for (int fi = 0; fi < mesh.nFaces; ++fi)
    {
        const Face& f = mesh.faces[fi];
        if (!isBoundaryFace(f)) continue;
        if (f.n1 == point_id || f.n2 == point_id)
        {
            const vector& p1 = mesh.points[f.n1];
            const vector& p2 = mesh.points[f.n2];
            s += norm2(p2.x - p1.x, p2.y - p1.y);
            ++cnt;
        }
    }
    if (cnt == 0) return 1.0;
    return s / (double)cnt;
}

double computeBoundaryObjective(
    dvmSolver& solver,
    const BoundaryFunctional& obj)
{
    const auto& mesh = solver.mesh;
    double J_local = 0.0;

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        if (f.bc_type != BCType::wall &&
            f.bc_type != BCType::pressure_far_field) {
            continue;
        }

        double A = f.Sf.mag();
        vector nf = f.Sf / A;
        int ghost = f.neigh;

        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x * solver.Vx[vi] + nf.y * solver.Vy[vi];
            int idx_g = solver.index_vdf(ghost, vi);

            double h = solver.vdf[idx_g];
            double m = (cn > 0.0) ? obj.mPlus(solver, facei, vi)
                                  : obj.mMinus(solver, facei, vi);

            J_local += A * cn * m * h * solver.feq[vi] * solver.weight[vi];
        }
    }

    double J_global = 0.0;
    MPI_Allreduce(&J_local, &J_global, 1, MPI_DOUBLE, MPI_SUM, solver.comm);
    return J_global;
}

/* ============================================================
 * 目标函数
 * J = 扰动点附近 owner 单元的平均宏观量
 * macro_comp: 0=rho, 1=ux, 2=uy
 * ============================================================ */

double runPrimalAndEvalJ(const Mesh& mesh,
                         const SolverConfig& cfg,
                         MPI_Comm comm,
                         int point_id)
{
    dvmSolver solver(mesh, cfg, comm);

    for (int iter = 0; iter < cfg.max_iter; ++iter)
    {
        solver.step(iter);

        if (solver.res_rho < cfg.tol &&
            solver.res_ux  < cfg.tol &&
            solver.res_uy  < cfg.tol)
        {
            break;
        }
    }

    XForceFunctional obj;
    return computeBoundaryObjective(solver, obj);

}

/* ============================================================
 * 单点扰动
 * ============================================================ */

void perturbOnePoint(Mesh& mesh, int point_id, int coord, double ds)
{
    if (coord == 0) mesh.points[point_id].x += ds;
    else            mesh.points[point_id].y += ds;
}

/* ============================================================
 * FD 主流程
 * ============================================================ */

bool validateOneBoundaryPoint(const Mesh& base_mesh,
                        const SolverConfig& cfg,
                        MPI_Comm comm)
{

    int validate_point = 2550;
    int validate_coord = 1; // 0=x, 1=y
    double eps_scale = 1e-4;


    const double href = localReferenceLength(base_mesh, validate_point);
    const double eps = eps_scale * href;

    // ADJ
    double g_adj = 0.0;
    dvmSolver solver(base_mesh, cfg, comm);

    for (int iter = 1; iter < cfg.max_iter; ++iter)
    {
        solver.step(iter);

        if (solver.res_rho < cfg.tol &&
            solver.res_ux  < cfg.tol &&
            solver.res_uy  < cfg.tol)
        {
            break;
        }
    }

    solver.initialAdj();

    for (int iter = 1; iter < cfg.max_iter; ++iter)
    {
        solver.stepAdj(iter);

        if (solver.res_arho < cfg.tol &&
            solver.res_aux  < cfg.tol &&
            solver.res_auy  < cfg.tol)
        {
            break;
        }
    }

    XForceFunctional obj;
    BoundarySensitivityAssembler assembler;

    std::vector<FaceGeomGrad> faceGrad;
    std::vector<NodeGrad> nodeGrad;

    assembler.assembleFaceGradients(solver, obj, faceGrad);
    assembler.accumulateNodeGradients(base_mesh, faceGrad, nodeGrad);

    double J0 = computeBoundaryObjective(solver, obj);
    g_adj = (validate_coord == 0) ? nodeGrad[validate_point].dx : nodeGrad[validate_point].dy;

    // FD
    Mesh plus_mesh = base_mesh;
    Mesh minus_mesh = base_mesh;

    perturbOnePoint(plus_mesh,  validate_point, validate_coord, +eps);
    perturbOnePoint(minus_mesh, validate_point, validate_coord, -eps);

    computeGeometry(plus_mesh, comm);
    computeGeometry(minus_mesh, comm);

    const double Jp = runPrimalAndEvalJ(plus_mesh,  cfg, comm, validate_point);
    const double Jm = runPrimalAndEvalJ(minus_mesh, cfg, comm, validate_point);
    const double g_fd   = (Jp - Jm) / (2.0 * eps);

    // compare with adjoint gradient
    const double rel_err = std::fabs(g_fd - g_adj) / g_fd;
    const double abs_err = std::fabs(g_fd - g_adj);

    std::printf(
        "[validation] point=%d coord=%c eps=%.6e\n"
        "  J0   = %.12e\n"
        "  J+   = %.12e\n"
        "  J-   = %.12e\n"
        "  FD   = %.12e\n"
        "  ADJ  = %.12e\n"
        "  REL  = %.12e\n"
        "  ABS  = %.12e\n",
        validate_point,
        (validate_coord == 0 ? 'x' : 'y'),
        eps,
        J0, Jp, Jm, g_fd, g_adj, rel_err, abs_err);

    return true;
}