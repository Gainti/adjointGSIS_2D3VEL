#include "boundary_sensitivity.h"
#include "defs.h"
#include <cmath>
#include <stdexcept>
#include <array>

// =========================
// objective
// =========================
double XForceFunctional::mPlus(const dvmSolver& solver, int facei, int vi) const
{
    const auto& mesh=solver.mesh;
    if(mesh.faces[facei].bc_type==BCType::pressure_far_field){
        return 2.0 * solver.Vx[vi];
    }
    return 0.0;

}

double XForceFunctional::mMinus(const dvmSolver& solver, int facei, int vi) const
{
    const auto& mesh=solver.mesh;
    if(mesh.faces[facei].bc_type==BCType::pressure_far_field){
        return 2.0 * solver.Vx[vi];
    }
    return 0.0;
}

// =========================
// geometry derivatives
// =========================
EdgeGeomDeriv2D computeEdgeGeomDeriv2D(const vector& r1, const vector& r2)
{
    EdgeGeomDeriv2D g;

    double dx = r2.x - r1.x;
    double dy = r2.y - r1.y;

    double A = std::sqrt(dx*dx + dy*dy);
    if (A <= 1e-30) {
        throw std::runtime_error("degenerate boundary edge");
    }

    g.A = A;
    g.C = vector(0.5*(r1.x+r2.x), 0.5*(r1.y+r2.y), 0.0);
    g.n = vector(dy/A, -dx/A, 0.0);

    g.dA_dr1 = {-dx/A, -dy/A};
    g.dA_dr2 = { dx/A,  dy/A};

    // dC/dr
    g.dCdr[0][0][0] = 0.5; g.dCdr[0][0][1] = 0.0;
    g.dCdr[0][1][0] = 0.0; g.dCdr[0][1][1] = 0.5;
    g.dCdr[1][0][0] = 0.5; g.dCdr[1][0][1] = 0.0;
    g.dCdr[1][1][0] = 0.0; g.dCdr[1][1][1] = 0.5;

    double A3 = A*A*A;

    // dn / dr1x
    g.dndr[0][0][0] =  dx*dy / A3;
    g.dndr[0][1][0] = -dy*dy / A3;

    // dn / dr1y
    g.dndr[0][0][1] = -dx*dx / A3;
    g.dndr[0][1][1] = -dx*dy / A3;

    // dr2 = -dr1
    g.dndr[1][0][0] = -g.dndr[0][0][0];
    g.dndr[1][1][0] = -g.dndr[0][1][0];
    g.dndr[1][0][1] = -g.dndr[0][0][1];
    g.dndr[1][1][1] = -g.dndr[0][1][1];

    return g;
}

// =========================
// diffuse boundary model
// =========================
DiffuseBoundaryModel::KernelGrad
DiffuseBoundaryModel::computeKernelAndGradN(
    dvmSolver& solver, int facei, int owner)
{
    KernelGrad out;
    const auto& f = solver.mesh.faces[facei];
    vector nf = f.Sf / f.Sf.mag();

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x * solver.Vx[vi] + nf.y * solver.Vy[vi];
        if (cn > 0.0) {
            int idx_owner = solver.index_vdf(owner, vi);
            double hp = solver.vdf[idx_owner];
            double coeff = 2.0 / PI * solver.exp_c2[vi] * hp * solver.weight[vi];

            out.K += coeff * cn;
            out.dKdn[0] += coeff * solver.Vx[vi];
            out.dKdn[1] += coeff * solver.Vy[vi];
        }
    }

    return out;
}

std::array<double,2> DiffuseBoundaryModel::wallVelocity(
    dvmSolver& solver, int facei)
{
    const auto& f = solver.mesh.faces[facei];
    vector nf = f.Sf / f.Sf.mag();

    std::array<double,2> uw{0.0, 0.0};

    if (f.bc_type == BCType::pressure_far_field) {
        switch (solver.cfg.uwall) {
            case 0:
                uw[0] = nf.y;
                uw[1] = -nf.x;
                break;
            case 1:
                uw[0] = 1.0;
                uw[1] = 0.0;
                break;
            case 2:
                uw[0] = 0.0;
                uw[1] = 1.0;
                break;
            default:
                break;
        }
    }

    return uw;
}

double DiffuseBoundaryModel::hb(
    dvmSolver& solver, int facei, int vi)
{
    const auto& f = solver.mesh.faces[facei];
    vector nf = f.Sf / f.Sf.mag();
    auto uw = wallVelocity(solver, facei);

    double uwn   = uw[0]*nf.x + uw[1]*nf.y;
    double udotv = uw[0]*solver.Vx[vi] + uw[1]*solver.Vy[vi];

    return 2.0 * udotv - sqrtPI * uwn;
}

std::array<double,2> DiffuseBoundaryModel::dhb_dn(
    dvmSolver& solver, int facei, int vi)
{
    const auto& f = solver.mesh.faces[facei];
    vector nf = f.Sf / f.Sf.mag();

    auto uw = wallVelocity(solver, facei);

    std::array<double,2> duw_dn_x{0.0, 0.0};
    std::array<double,2> duw_dn_y{0.0, 0.0};

    if (f.bc_type == BCType::pressure_far_field && solver.cfg.uwall == 0) {
        // uw = [ny, -nx]
        duw_dn_x = {0.0, -1.0};
        duw_dn_y = {1.0,  0.0};
    }

    std::array<double,2> out{0.0, 0.0};

    out[0] =
        2.0 * (duw_dn_x[0]*solver.Vx[vi] + duw_dn_x[1]*solver.Vy[vi])
        - sqrtPI * ((duw_dn_x[0]*nf.x + duw_dn_x[1]*nf.y) + uw[0]);

    out[1] =
        2.0 * (duw_dn_y[0]*solver.Vx[vi] + duw_dn_y[1]*solver.Vy[vi])
        - sqrtPI * ((duw_dn_y[0]*nf.x + duw_dn_y[1]*nf.y) + uw[1]);

    return out;
}

// =========================
// owner-cell gradient only
// =========================
void BoundarySensitivityAssembler::computeOwnerCellGradient(
    dvmSolver& solver,
    int owner,
    std::vector<double>& gradHx,
    std::vector<double>& gradHy)
{
    const auto& mesh  = solver.mesh;
    const auto& cell  = mesh.cells[owner];
    const auto& faces = mesh.faces;

    gradHx.assign(solver.Nv, 0.0);
    gradHy.assign(solver.Nv, 0.0);

    double V = cell.V;
    if (V <= 1e-30) return;

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double gx = 0.0;
        double gy = 0.0;

        for (int facej : cell.cell2face) {
            const auto& fj = faces[facej];
            int ownerj = fj.owner;
            int neighj = fj.neigh;

            int idx_owner = solver.index_vdf(ownerj, vi);
            int idx_neigh = solver.index_vdf(neighj, vi);

            double hf = 0.5 * (solver.vdf[idx_owner] + solver.vdf[idx_neigh]);

            if (owner == ownerj) {
                gx += hf * fj.Sf.x;
                gy += hf * fj.Sf.y;
            } else if (owner == neighj) {
                gx -= hf * fj.Sf.x;
                gy -= hf * fj.Sf.y;
            }
        }

        gradHx[vi] = gx / V;
        gradHy[vi] = gy / V;
    }
}

// =========================
// dJ/dA, dJ/dn
// =========================
void BoundarySensitivityAssembler::accumulate_dJdA_dJdn(
    dvmSolver& solver,
    int facei,
    const BoundaryFunctional& obj,
    FaceGeomGrad& g)
{
    const auto& f = solver.mesh.faces[facei];

    int ghost = f.neigh;
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
        int idx_g = solver.index_vdf(ghost, vi);

        double h = solver.vdf[idx_g];
        double m = (cn > 0.0) ? obj.mPlus(solver, facei, vi)
                              : obj.mMinus(solver, facei, vi);

        double w = solver.feq[vi] * solver.weight[vi];

        g.dLdA    += cn * m * h * w;
        g.dLdn[0] += A * m * h * solver.Vx[vi] * w;
        g.dLdn[1] += A * m * h * solver.Vy[vi] * w;
    }
}

// =========================
// dJ/dC with Gauss quadrature
// =========================
void BoundarySensitivityAssembler::accumulate_dJdC(
    dvmSolver& solver,
    int facei,
    const BoundaryFunctional& obj,
    const std::vector<double>& gradHx,
    const std::vector<double>& gradHy,
    FaceGeomGrad& g)
{
    const auto& mesh = solver.mesh;
    const auto& f = mesh.faces[facei];

    const vector& r1 = mesh.points[f.n1];
    const vector& r2 = mesh.points[f.n2];

    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
        double m  = (cn > 0.0) ? obj.mPlus(solver, facei, vi)
                               : obj.mMinus(solver, facei, vi);

        double coeff = A*cn*m*solver.feq[vi] * solver.weight[vi];

        g.dLdC[0] += coeff * gradHx[vi];
        g.dLdC[1] += coeff * gradHy[vi];
    }
}

// =========================
// dKdC
// =========================
std::array<double,2> BoundarySensitivityAssembler::compute_dKdC(
    dvmSolver& solver,
    int facei,
    int owner,
    const std::vector<double>& gradHx,
    const std::vector<double>& gradHy)
{
    const auto& f = solver.mesh.faces[facei];
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    std::array<double,2> dKdC{0.0, 0.0};

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
        if (cn <= 0.0) continue;

        double coeff = 2.0 / PI * cn * solver.exp_c2[vi] * solver.weight[vi];

        dKdC[0] += coeff * gradHx[vi];
        dKdC[1] += coeff * gradHy[vi];
    }

    return dKdC;
}

// =========================
// dBw/dn
// =========================
void BoundarySensitivityAssembler::accumulate_dBwdn(
    dvmSolver& solver,
    int facei,
    int owner,
    const BoundaryFunctional& obj,
    FaceGeomGrad& g)
{
    const auto& f = solver.mesh.faces[facei];
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    auto Kg = DiffuseBoundaryModel::computeKernelAndGradN(solver, facei, owner);

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
        if (cn >= 0.0) continue;

        int idx_owner = solver.index_vdf(owner, vi);
        double phi_minus = solver.avdf[idx_owner];
        double mminus = obj.mMinus(solver, facei, vi);
        double lambda = -cn * (phi_minus + mminus);

        double w = solver.feq[vi] * solver.weight[vi];
        auto dhbdn = DiffuseBoundaryModel::dhb_dn(solver, facei, vi);

        g.dLdn[0] += A * lambda * (-Kg.dKdn[0] - dhbdn[0]) * w;
        g.dLdn[1] += A * lambda * (-Kg.dKdn[1] - dhbdn[1]) * w;
    }
}

// =========================
// dBw/dC
// =========================
void BoundarySensitivityAssembler::accumulate_dBwdC(
    dvmSolver& solver,
    int facei,
    int owner,
    const BoundaryFunctional& obj,
    const std::vector<double>& gradHx,
    const std::vector<double>& gradHy,
    const std::array<double,2>& dKdC,
    FaceGeomGrad& g)
{
    (void)owner;

    const auto& f = solver.mesh.faces[facei];
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
        if (cn >= 0.0) continue;

        int idx_owner = solver.index_vdf(f.owner, vi);
        double phi_minus = solver.avdf[idx_owner];
        double mminus = obj.mMinus(solver, facei, vi);
        double lambda = -cn * (phi_minus + mminus);

        double w = solver.feq[vi] * solver.weight[vi];

        // current approximation:
        // dh^-/dC ~ grad(owner)
        // dhb/dC = 0
        g.dLdC[0] += A * lambda * (gradHx[vi] - dKdC[0]) * w;
        g.dLdC[1] += A * lambda * (gradHy[vi] - dKdC[1]) * w;
    }
}

// =========================
// face assembly
// =========================
void BoundarySensitivityAssembler::assembleFaceGradients(
    dvmSolver& solver,
    const BoundaryFunctional& obj,
    std::vector<FaceGeomGrad>& faceGrad)
{
    const auto& mesh = solver.mesh;
    faceGrad.assign(mesh.nFaces, FaceGeomGrad{});

    std::vector<double> ownerGradHx;
    std::vector<double> ownerGradHy;

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        if (f.bc_type != BCType::wall &&
            f.bc_type != BCType::pressure_far_field) {
            continue;
        }

        int owner = f.owner;

        FaceGeomGrad g;

        // 1) owner cell gradient for this face only
        computeOwnerCellGradient(solver, owner, ownerGradHx, ownerGradHy);

        // 2) objective terms
        accumulate_dJdA_dJdn(solver, facei, obj, g);
        accumulate_dJdC(solver, facei, obj, ownerGradHx, ownerGradHy, g);

        // 3) boundary constraint terms
        auto dKdC = compute_dKdC(
            solver, facei, owner, ownerGradHx, ownerGradHy);

        accumulate_dBwdn(solver, facei, owner, obj, g);
        accumulate_dBwdC(
            solver, facei, owner, obj,
            ownerGradHx, ownerGradHy, dKdC, g);

        faceGrad[facei] = g;
    }
}

// =========================
// node accumulation
// =========================
void BoundarySensitivityAssembler::accumulateNodeGradients(
    const Mesh& mesh,
    const std::vector<FaceGeomGrad>& faceGrad,
    std::vector<NodeGrad>& nodeGrad)
{
    nodeGrad.assign(mesh.points.size(), NodeGrad{});

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        int n1 = f.n1;
        int n2 = f.n2;

        auto gd = computeEdgeGeomDeriv2D(mesh.points[n1], mesh.points[n2]);
        const auto& fg = faceGrad[facei];

        int nodes[2] = {n1, n2};

        for (int s = 0; s < 2; ++s) {
            NodeGrad add;

            const auto& dAdr = (s == 0) ? gd.dA_dr1 : gd.dA_dr2;
            add.dx += fg.dLdA * dAdr[0];
            add.dy += fg.dLdA * dAdr[1];

            add.dx += fg.dLdC[0] * gd.dCdr[s][0][0]
                    + fg.dLdC[1] * gd.dCdr[s][1][0];
            add.dy += fg.dLdC[0] * gd.dCdr[s][0][1]
                    + fg.dLdC[1] * gd.dCdr[s][1][1];

            add.dx += fg.dLdn[0] * gd.dndr[s][0][0]
                    + fg.dLdn[1] * gd.dndr[s][1][0];
            add.dy += fg.dLdn[0] * gd.dndr[s][0][1]
                    + fg.dLdn[1] * gd.dndr[s][1][1];

            nodeGrad[nodes[s]].dx += add.dx;
            nodeGrad[nodes[s]].dy += add.dy;
        }
    }
}