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
    int point_id, int coord, double eps_scale,
    MPI_Comm comm)
{
    // ADJ
    double g_adj = 0.0;
    dvmSolver solver(base_mesh, vel, cfg, comm);
    // for (int iter = 1; iter < cfg.max_iter; ++iter)
    // {
    //     solver.step(iter);

    //     if (solver.res_rho < cfg.tol &&
    //         solver.res_ux  < cfg.tol &&
    //         solver.res_uy  < cfg.tol)
    //     {
    //         break;
    //     }
    // }
    double J0 = computeObjective(solver);

    // adjointDVM adj(solver);
    // for (int iter = 1; iter < cfg.max_iter; ++iter)
    // {
    //     adj.step(iter);

    //     if (adj.res_arho < cfg.tol &&
    //         adj.res_aux  < cfg.tol &&
    //         adj.res_auy  < cfg.tol)
    //     {
    //         break;
    //     }
    // }
    // 计算伴随导数
    // BoundarySensitivityAssembler assembler;
    // std::vector<FaceGeomGrad> faceGrad;
    // std::vector<NodeGrad> nodeGrad;
    // assembler.assembleFaceGradients(solver, adj, faceGrad);
    // assembler.accumulateNodeGradients(base_mesh, faceGrad, nodeGrad);

    // g_adj = (coord == 0) ? nodeGrad[point_id].dx : nodeGrad[point_id].dy;

    // std::vector<char> isBoundary(base_mesh.points.size(), 0);
    // for (int facei = base_mesh.nInternalFaces; facei < base_mesh.nFaces; ++facei) {
    //     const auto& f = base_mesh.faces[facei];
    //     if(f.bc_type == BCType::internal) continue;
    //     isBoundary[f.n1] = 1;
    //     isBoundary[f.n2] = 1;
    // }
    // for(int pointI=0; pointI<base_mesh.points.size(); ++pointI){
    //     if(isBoundary[pointI]){
    //         printf("point %d: x = %e, y= %e, dJ/dx = %e, dJ/dy = %e\n", pointI, base_mesh.points[pointI].x, base_mesh.points[pointI].y, nodeGrad[pointI].dx, nodeGrad[pointI].dy);
    //     }
    // }

    // FD
    Mesh plus_mesh = base_mesh;
    Mesh minus_mesh = base_mesh;

    const double href = localReferenceLength(base_mesh, point_id);
    const double eps = eps_scale * href;

    perturbOnePoint(plus_mesh,  point_id, coord, +eps);
    perturbOnePoint(minus_mesh, point_id, coord, -eps);

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
        point_id,
        (coord == 0 ? 'x' : 'y'),
        eps,
        J0, Jp, Jm, g_fd, g_adj, rel_err, abs_err);

    return true;
}


void perturbYscaling(Mesh& mesh, double scale)
{
    std::vector<char> isBoundary(mesh.points.size(), 0);
    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        if(f.bc_type == BCType::pressure_far_field){
            isBoundary[f.n1] = 1;
            isBoundary[f.n2] = 1;
        }
    }
    for(int pointI=0; pointI<mesh.points.size(); ++pointI){
        if(isBoundary[pointI]){
            mesh.points[pointI].y *= scale;
        }
    }
    computeGeometry(mesh, MPI_COMM_WORLD);
}

double projectGradientToYScalingDirection(
    const Mesh& mesh,
    const std::vector<NodeGrad>& nodeGrad,
    MPI_Comm comm)
{
    std::vector<char> isBoundary(mesh.points.size(), 0);
    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        if(f.bc_type == BCType::pressure_far_field){
            isBoundary[f.n1] = 1;
            isBoundary[f.n2] = 1;
        }
    }

    double local = 0.0;
    for (size_t pid = 0; pid < mesh.points.size(); ++pid) {
        if (!isBoundary[pid]) continue;
        // local += nodeGrad[pid].dy * mesh.points[pid].y;
        local += nodeGrad[pid].dy;
    }

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

bool validateYscaleboundary(const Mesh& globalMesh,
    const Mesh& localMesh,
    const VelocitySpace& vel,
    const SolverConfig& cfg,
    MPI_Comm comm)
{
    double eps_scale = 1e-7;

    // ADJ
    double g_adj = 0.0;
    dvmSolver solver(localMesh, vel, cfg, comm);
    // for (int iter = 1; iter < cfg.max_iter; ++iter)
    // {
    //     solver.step(iter);

    //     if (solver.res_rho < cfg.tol &&
    //         solver.res_ux  < cfg.tol &&
    //         solver.res_uy  < cfg.tol)
    //     {
    //         break;
    //     }
    // }
    double J0 = computeObjective(solver);

    // adjointDVM adj(solver);
    // for (int iter = 1; iter < cfg.max_iter; ++iter)
    // {
    //     adj.step(iter);

    //     if (adj.res_arho < cfg.tol &&
    //         adj.res_aux  < cfg.tol &&
    //         adj.res_auy  < cfg.tol)
    //     {
    //         break;
    //     }
    // }
    // BoundarySensitivityAssembler assembler;
    // std::vector<FaceGeomGrad> faceGrad;
    // std::vector<NodeGrad> nodeGrad;

    // assembler.assembleFaceGradients(solver, adj, faceGrad);
    // assembler.accumulateNodeGradients(localMesh, faceGrad, nodeGrad);
    // g_adj = projectGradientToYScalingDirection(localMesh, nodeGrad, comm);
    // printf("Adjoint gradient projected to y-scaling direction: %e\n", g_adj);

    // FD
    Mesh plus_mesh = localMesh;
    Mesh minus_mesh = localMesh;

    const double href = 1.0;
    const double eps = eps_scale * href;

    perturbYscaling(plus_mesh, 1.0 + eps);
    perturbYscaling(minus_mesh, 1.0 - eps);

    const double Jp = runPrimalAndEvalJ(plus_mesh, vel, cfg, comm);
    const double Jm = runPrimalAndEvalJ(minus_mesh, vel, cfg, comm);
    const double g_fd   = (Jp - Jm) / (2.0 * eps);

    // compare with adjoint gradient
    const double rel_err = std::fabs(g_fd - g_adj) / g_fd;
    const double abs_err = std::fabs(g_fd - g_adj);

    std::printf(
        "[validation] eps=%.6e\n"
        "  J0   = %.12e\n"
        "  J+   = %.12e\n"
        "  J-   = %.12e\n"
        "  FD   = %.12e\n"
        "  ADJ  = %.12e\n"
        "  REL  = %.12e\n"
        "  ABS  = %.12e\n",
        eps,
        J0, Jp, Jm, g_fd, g_adj, rel_err, abs_err);

    return true;
}

// 边界形函数扰动
static double boundaryShapePhi(double s, double L)
{
    if (L <= 0.0) return 0.0;
    const double xi = PI * s / L;
    const double t = std::sin(xi);
    return t * t;   // sin^2(pi s / L)
}
// 提取边界点构成单连通边界链
static bool buildPressureFarFieldPointChain(
    const Mesh& mesh,
    std::vector<int>& ordered_points,
    std::vector<double>& arc_s,
    double& total_length)
{
    ordered_points.clear();
    arc_s.clear();
    total_length = 0.0;

    const int nPoints = (int)mesh.points.size();

    std::vector<std::vector<int> > adj(nPoints);
    std::vector<char> is_pf_point(nPoints, 0);

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei)
    {
        const Face& f = mesh.faces[facei];
        if (f.bc_type != BCType::pressure_far_field) continue;

        const int a = f.n1;
        const int b = f.n2;

        adj[a].push_back(b);
        adj[b].push_back(a);

        is_pf_point[a] = 1;
        is_pf_point[b] = 1;
    }

    // 找起点：优先找度为1的端点
    int start = -1;
    for (int pid = 0; pid < nPoints; ++pid)
    {
        if (!is_pf_point[pid]) continue;
        if ((int)adj[pid].size() == 1)
        {
            start = pid;
            break;
        }
    }

    // 若没有端点，则可能是闭合边界，找任意一个边界点
    if (start < 0)
    {
        for (int pid = 0; pid < nPoints; ++pid)
        {
            if (is_pf_point[pid])
            {
                start = pid;
                break;
            }
        }
    }

    if (start < 0) return false;

    std::vector<char> visited(nPoints, 0);
    ordered_points.push_back(start);
    visited[start] = 1;

    int prev = -1;
    int curr = start;

    while (true)
    {
        int next = -1;
        for (size_t k = 0; k < adj[curr].size(); ++k)
        {
            const int cand = adj[curr][k];
            if (cand == prev) continue;
            if (visited[cand]) continue;
            next = cand;
            break;
        }

        if (next < 0) break;

        ordered_points.push_back(next);
        visited[next] = 1;
        prev = curr;
        curr = next;
    }

    if (ordered_points.size() < 2) return false;

    arc_s.resize(ordered_points.size(), 0.0);
    total_length = 0.0;

    for (size_t i = 1; i < ordered_points.size(); ++i)
    {
        const vector& p0 = mesh.points[ordered_points[i - 1]];
        const vector& p1 = mesh.points[ordered_points[i]];
        total_length += norm2(p1.x - p0.x, p1.y - p0.y);
        arc_s[i] = total_length;
    }

    return true;
}
void perturbPressureFarFieldBoundaryY(
    Mesh& mesh,
    double alpha)
{
    std::vector<int> ordered_points;
    std::vector<double> arc_s;
    double total_length = 0.0;

    const bool ok = buildPressureFarFieldPointChain(
        mesh, ordered_points, arc_s, total_length);

    if (!ok)
    {
        printf("Failed to build pressure_far_field boundary chain.\n");
        return;
    }

    for (size_t k = 0; k < ordered_points.size(); ++k)
    {
        const int pid = ordered_points[k];
        const double phi = boundaryShapePhi(arc_s[k], total_length);
        mesh.points[pid].y += alpha * phi;
    }

    computeGeometry(mesh, MPI_COMM_WORLD);
}
double projectGradientToPressureFarFieldYMode(
    const Mesh& mesh,
    const std::vector<NodeGrad>& nodeGrad,
    MPI_Comm comm)
{
    std::vector<int> ordered_points;
    std::vector<double> arc_s;
    double total_length = 0.0;

    const bool ok = buildPressureFarFieldPointChain(
        mesh, ordered_points, arc_s, total_length);

    if (!ok)
    {
        printf("Failed to build pressure_far_field boundary chain.\n");
        return 0.0;
    }

    double local = 0.0;

    for (size_t k = 0; k < ordered_points.size(); ++k)
    {
        const int pid = ordered_points[k];
        const double phi = boundaryShapePhi(arc_s[k], total_length);
        local += nodeGrad[pid].dy * phi;
    }

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}
bool validatePressureFarFieldBoundaryYMode(
    const Mesh& localMesh,
    const VelocitySpace& vel,
    const SolverConfig& cfg,
    MPI_Comm comm)
{
    double eps_scale = 1e-6;

    // =========================
    // 1) primal
    // =========================
    dvmSolver solver(localMesh, vel, cfg, comm);

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

    double J0 = computeObjective(solver);

    // =========================
    // 2) adjoint
    // =========================
    adjointDVM adj(localMesh, vel, cfg, comm);

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

    // =========================
    // 3) node gradients
    // =========================
    BoundarySensitivityAssembler assembler;

    std::vector<FaceGeomGrad> faceGrad;
    std::vector<NodeGrad> nodeGrad;

    assembler.assembleFaceGradients(solver, adj, faceGrad);
    assembler.accumulateNodeGradients(localMesh, faceGrad, nodeGrad);

    // =========================
    // 4) build boundary chain and choose eps
    // =========================
    std::vector<int> ordered_points;
    std::vector<double> arc_s;
    double total_length = 0.0;

    const bool ok = buildPressureFarFieldPointChain(
        localMesh, ordered_points, arc_s, total_length);

    if (!ok)
    {
        printf("Failed to build pressure_far_field boundary chain.\n");
        return false;
    }

    const double eps = eps_scale * total_length;

    // =========================
    // 5) ADJ projection
    // =========================
    const double g_adj = projectGradientToPressureFarFieldYMode(
        localMesh, nodeGrad, comm);

    // =========================
    // 6) FD
    // =========================
    Mesh plus_mesh  = localMesh;
    Mesh minus_mesh = localMesh;

    perturbPressureFarFieldBoundaryY(plus_mesh,  +eps);
    perturbPressureFarFieldBoundaryY(minus_mesh, -eps);

    const double Jp = runPrimalAndEvalJ(plus_mesh,  vel, cfg, comm);
    const double Jm = runPrimalAndEvalJ(minus_mesh, vel, cfg, comm);

    const double g_fd = (Jp - Jm) / (2.0 * eps);

    const double denom = std::max(std::max(std::fabs(g_fd), std::fabs(g_adj)), 1e-14);
    const double rel_err = std::fabs(g_fd - g_adj) / denom;
    const double abs_err = std::fabs(g_fd - g_adj);

    std::printf(
        "[pressure-far-field y-mode validation]\n"
        "  nPts = %d\n"
        "  L    = %.12e\n"
        "  eps  = %.12e\n"
        "  J0   = %.12e\n"
        "  J+   = %.12e\n"
        "  J-   = %.12e\n"
        "  FD   = %.12e\n"
        "  ADJ  = %.12e\n"
        "  REL  = %.12e\n"
        "  ABS  = %.12e\n",
        (int)ordered_points.size(),
        total_length,
        eps,
        J0, Jp, Jm, g_fd, g_adj, rel_err, abs_err);

    return true;
}