#include "boundary_sensitivity.h"
#include "defs.h"
#include "mesh.h"
#include <cmath>
#include <stdexcept>
#include <array>

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
    g.dndr[0][1][0] =  dy*dy / A3;

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

    double V = cell.V;
    if (V <= 1e-30) return;

    for (int vi = 0; vi < solver.Nv; ++vi) {
        double gx = 0.0;
        double gy = 0.0;

        for (int facej : cell.cell2face) {
            const auto& fj = faces[facej];
            int ownerj = fj.owner;
            int neighj = fj.neigh;

            vector Sf = fj.Sf;
            double vn = Sf.x * solver.Vx[vi] + Sf.y * solver.Vy[vi];
            double hf = 0.0;
            if (vn > 0.0) {
                hf = solver.vdf[ownerj*solver.Nv + vi];
            }else{
                hf = solver.vdf[neighj*solver.Nv + vi];
            }

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
    FaceGeomGrad& g)
{
    const auto& f = solver.mesh.faces[facei];

    int neigh = f.neigh;
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    double dJdA = 0.0;
    double dJdn[2] = {0.0, 0.0};

    if(f.bc_type == BCType::pressure_far_field) {
        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
    
            double h = solver.vdf[neigh*solver.Nv + vi];
            double m = objectiveWeight(solver.Vx[vi], solver.Vy[vi]);

            double w = solver.feq[vi] * solver.weight[vi];
    
            dJdA += cn * m * h * w;
            dJdn[0] += A * m * h * solver.Vx[vi] * w;
            dJdn[1] += A * m * h * solver.Vy[vi] * w;
        }
    }

    g.dJdA = dJdA;
    g.dJdn[0] = dJdn[0];
    g.dJdn[1] = dJdn[1];
}

// =========================
// dJ/dC with Gauss quadrature
// =========================
void BoundarySensitivityAssembler::accumulate_dJdC(
    dvmSolver& solver,
    int facei,
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

    double dJdC[2] = {0.0, 0.0};

    if(f.bc_type == BCType::pressure_far_field){
        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
            double m  = objectiveWeight(solver.Vx[vi], solver.Vy[vi]);
    
            double coeff = A*cn*m*solver.feq[vi] * solver.weight[vi];
    
            dJdC[0] += coeff * gradHx[vi];
            dJdC[1] += coeff * gradHy[vi];
         }
    }

    g.dJdC[0] = dJdC[0];
    g.dJdC[1] = dJdC[1];
}

// =========================
// dBw/dn
// =========================
void BoundarySensitivityAssembler::accumulate_dBwdn(
    dvmSolver& solver,
    int facei,
    FaceGeomGrad& g)
{
    const auto& f = solver.mesh.faces[facei];
    int owner = f.owner;
    int neigh = f.neigh;
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    // compute dK/dn
    double dKdn[2] = {0.0, 0.0};
    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x * solver.Vx[vi] + nf.y * solver.Vy[vi];
        if (cn > 0.0) {
            double hp = solver.vdf[neigh*solver.Nv + vi];
            double coeff = 2.0 / PI * solver.exp_c2[vi] * hp * solver.weight[vi];

            dKdn[0] += coeff * solver.Vx[vi];
            dKdn[1] += coeff * solver.Vy[vi];
        }
    }

    double dBwdn[2] = {0.0, 0.0};
    if(f.bc_type == BCType::pressure_far_field) {
        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
            if (cn >= 0.0) continue;
    
            double phi = solver.avdf[neigh*solver.Nv + vi];
            double m = objectiveWeight(solver.Vx[vi], solver.Vy[vi]);
            double lambda = -cn * (phi + m);
    
            double w = solver.feq[vi] * solver.weight[vi];
            auto dhbdn = DiffuseBoundaryModel::dhb_dn(solver, facei, vi);
            dBwdn[0] += lambda * (-dKdn[0] - dhbdn[0]) * w;
            dBwdn[1] += lambda * (-dKdn[1] - dhbdn[1]) * w;
    
        }
    }else if(f.bc_type == BCType::wall) {
        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
            if (cn >= 0.0) continue;
    
            double phi_minus = solver.avdf[neigh*solver.Nv + vi];
            double lambda = -cn * phi_minus;
    
            double w = solver.feq[vi] * solver.weight[vi];
            auto dhbdn = DiffuseBoundaryModel::dhb_dn(solver, facei, vi);
            dBwdn[0] += lambda * (-dKdn[0] - dhbdn[0]) * w;
            dBwdn[1] += lambda * (-dKdn[1] - dhbdn[1]) * w;
    
        }
    }
    g.dBwdn[0] = A * dBwdn[0];
    g.dBwdn[1] = A * dBwdn[1];
}

// =========================
// dBw/dC
// =========================
void BoundarySensitivityAssembler::accumulate_dBwdC(
    dvmSolver& solver,
    int facei,
    const std::vector<double>& gradHx,
    const std::vector<double>& gradHy,
    FaceGeomGrad& g)
{
    const auto& f = solver.mesh.faces[facei];
    int owner = f.owner;
    int neigh = f.neigh;
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    // compute dK/dC
    double dKdC[2] = {0.0, 0.0};
    for (int vi = 0; vi < solver.Nv; ++vi) {
        double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
        if (cn <= 0.0) continue;

        double coeff = 2.0 / PI * cn * solver.exp_c2[vi] * solver.weight[vi];

        dKdC[0] += coeff * gradHx[vi];
        dKdC[1] += coeff * gradHy[vi];
    }

    // dBw/dC
    double dBwdC[2] = {0.0, 0.0};
    if(f.bc_type == BCType::pressure_far_field) {
        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
            if (cn >= 0.0) continue;
    
            double phi = solver.avdf[neigh*solver.Nv + vi];
            double m = objectiveWeight(solver.Vx[vi], solver.Vy[vi]);
            double lambda = -cn * (phi + m);
    
            double w = solver.feq[vi] * solver.weight[vi];
    
            dBwdC[0] += lambda * (gradHx[vi] - dKdC[0]) * w;
            dBwdC[1] += lambda * (gradHy[vi] - dKdC[1]) * w;
        }
    }else if(f.bc_type == BCType::wall) {
        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x*solver.Vx[vi] + nf.y*solver.Vy[vi];
            if (cn >= 0.0) continue;
    
            double phi = solver.avdf[neigh*solver.Nv + vi];
            double lambda = -cn * phi;
    
            double w = solver.feq[vi] * solver.weight[vi];
    
            dBwdC[0] += lambda * (gradHx[vi] - dKdC[0]) * w;
            dBwdC[1] += lambda * (gradHy[vi] - dKdC[1]) * w;
        }
    }
    g.dBwdC[0] = A * dBwdC[0];
    g.dBwdC[1] = A * dBwdC[1];
}

// =========================
// face assembly
// =========================
void BoundarySensitivityAssembler::assembleFaceGradients(
    dvmSolver& solver,
    std::vector<FaceGeomGrad>& faceGrad)
{
    const auto& mesh = solver.mesh;
    faceGrad.assign(mesh.nFaces, FaceGeomGrad{});

    std::vector<double> ownerGradHx;
    std::vector<double> ownerGradHy;

    ownerGradHx.resize(solver.Nv, 0.0);
    ownerGradHy.resize(solver.Nv, 0.0);

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        if (f.bc_type != BCType::wall &&
            f.bc_type != BCType::pressure_far_field) {
            continue;
        }

        int owner = f.owner;

        FaceGeomGrad g;

        // compute gradient of owner cell
        computeOwnerCellGradient(solver, owner, ownerGradHx, ownerGradHy);

        // objective terms
        accumulate_dJdA_dJdn(solver, facei, g);
        accumulate_dJdC(solver, facei, ownerGradHx, ownerGradHy, g);

        // boundary constraint terms
        accumulate_dBwdn(solver, facei, g);
        accumulate_dBwdC(solver, facei, ownerGradHx, ownerGradHy, g);

        // sum
        g.dLdA = g.dJdA;
        g.dLdC[0] = g.dJdC[0] + g.dBwdC[0];
        g.dLdC[1] = g.dJdC[1] + g.dBwdC[1];
        g.dLdn[0] = g.dJdn[0] + g.dBwdn[0];
        g.dLdn[1] = g.dJdn[1] + g.dBwdn[1];

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

        double magSf = f.Sf.mag();
        if(std::abs(gd.n.x-f.Sf.x/magSf)>1e-8 || std::abs(gd.n.y-f.Sf.y/magSf)>1e-8){
            printf("Error: edge geom deriv doesn't match face normal\n");
        }

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