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

double runPrimalAndEvalJ(const Mesh& mesh,
                        const VelocitySpace& vel,
                        const SolverConfig& cfg,
                        MPI_Comm comm)
{
    dvmSolver solver(mesh, vel, cfg, comm);

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
    return computeObjective(solver);

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
                        const VelocitySpace& vel,
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
    dvmSolver solver(base_mesh, vel, cfg, comm);

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

    adjointDVM adj(solver);

    for (int iter = 1; iter < cfg.max_iter; ++iter)
    {
        adj.step(iter);

        if (adj.res_arho < cfg.tol &&
            adj.res_aux  < cfg.tol &&
            adj.res_auy  < cfg.tol)
        {
            break;
        }
    }

    BoundarySensitivityAssembler assembler;

    std::vector<FaceGeomGrad> faceGrad;
    std::vector<NodeGrad> nodeGrad;

    assembler.assembleFaceGradients(solver, adj, faceGrad);
    assembler.accumulateNodeGradients(base_mesh, faceGrad, nodeGrad);

    double J0 = computeObjective(solver);
    g_adj = (validate_coord == 0) ? nodeGrad[validate_point].dx : nodeGrad[validate_point].dy;

    // FD
    Mesh plus_mesh = base_mesh;
    Mesh minus_mesh = base_mesh;

    perturbOnePoint(plus_mesh,  validate_point, validate_coord, +eps);
    perturbOnePoint(minus_mesh, validate_point, validate_coord, -eps);

    computeGeometry(plus_mesh, comm);
    computeGeometry(minus_mesh, comm);

    const double Jp = runPrimalAndEvalJ(plus_mesh, vel, cfg, comm);
    const double Jm = runPrimalAndEvalJ(minus_mesh, vel, cfg, comm);
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