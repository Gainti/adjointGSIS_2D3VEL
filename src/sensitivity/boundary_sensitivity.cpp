#include "boundary_sensitivity.h"
#include "defs.h"
#include "mesh.h"
#include <cmath>
#include <stdexcept>
#include <array>


inline int index_grad(int vi, int i, int j) {
    return vi*Nvdf*dim + i*dim + j;
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
    dvmSolver& primal, int facei)
{
    const auto& f = primal.mesh.faces[facei];
    vector nf = f.Sf / f.Sf.mag();

    std::array<double,2> uw{0.0, 0.0};

    if (f.bc_type == BCType::pressure_far_field) {
        switch (primal.cfg.uwall) {
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
    dvmSolver& primal, int facei, int vi)
{
    const auto& f = primal.mesh.faces[facei];
    vector nf = f.Sf / f.Sf.mag();
    auto uw = wallVelocity(primal, facei);

    double uwn   = uw[0]*nf.x + uw[1]*nf.y;
    double udotv = uw[0]*primal.Vx[vi] + uw[1]*primal.Vy[vi];

    return 2.0 * udotv - sqrtPI * uwn;
}

std::array<double,2> DiffuseBoundaryModel::dhb_dn(
    dvmSolver& primal, int facei, int vi)
{
    const auto& f = primal.mesh.faces[facei];
    vector nf = f.Sf / f.Sf.mag();

    auto uw = wallVelocity(primal, facei);

    std::array<double,2> duw_dn_x{0.0, 0.0};
    std::array<double,2> duw_dn_y{0.0, 0.0};

    if (f.bc_type == BCType::pressure_far_field && primal.cfg.uwall == 0) {
        // uw = [ny, -nx]
        duw_dn_x = {0.0, -1.0};
        duw_dn_y = {1.0,  0.0};
    }

    std::array<double,2> out{0.0, 0.0};

    out[0] =
        2.0 * (duw_dn_x[0]*primal.Vx[vi] + duw_dn_x[1]*primal.Vy[vi])
        - sqrtPI * ((duw_dn_x[0]*nf.x + duw_dn_x[1]*nf.y) + uw[0]);

    out[1] =
        2.0 * (duw_dn_y[0]*primal.Vx[vi] + duw_dn_y[1]*primal.Vy[vi])
        - sqrtPI * ((duw_dn_y[0]*nf.x + duw_dn_y[1]*nf.y) + uw[1]);

    return out;
}

// Guass formula to compute dhdC on boundary face(using zero gradient)
void BoundarySensitivityAssembler::computeOwnerCellGradient(
    dvmSolver& primal,
    int cellI,
    std::vector<scalar>& gradh)
{
    const auto& mesh  = primal.mesh;
    const auto& cell  = mesh.cells[cellI];
    const auto& cells = mesh.cells;
    const auto& faces = mesh.faces;

    const auto& h =primal.vdf;
    const int Nv = primal.Nv;

    double V = cell.V;
    if (V <= 1e-30) return;

    for (int vi = 0; vi < Nv; ++vi) {
        scalar gradh_t[dim*Nvdf] = {Zero};

        for (int faceI : cell.cell2face) {
            const auto& fj = faces[faceI];
            int owner = fj.owner;
            int neigh = fj.neigh;

            vector Sf = fj.Sf;
            double vn = Sf.x * primal.Vx[vi] + Sf.y * primal.Vy[vi];

            vector xof = faces[faceI].Cf -cells[owner].C;
            vector xnf = faces[faceI].Cf -cells[neigh].C;

            scalar hf[Nvdf] = {Zero};
            scalar dh[dim*Nvdf] = {Zero};
            int idx_owner = primal.index_vdf(owner, vi);
            int idx_neigh = primal.index_vdf(neigh, vi);
            if (vn > 0.0) {
                primal.grad(owner, vi, dh, Nvdf);
                hf[0] = h[idx_owner+0] + dh[0]*xof.x + dh[1]*xof.y;
                hf[1] = h[idx_owner+1] + dh[2]*xof.x + dh[3]*xof.y;
            }else{
                primal.grad(neigh, vi, dh, Nvdf);
                hf[0] = h[idx_neigh+0] + dh[0]*xnf.x + dh[1]*xnf.y;
                hf[1] = h[idx_neigh+1] + dh[2]*xnf.x + dh[3]*xnf.y;
            }

            if (cellI == owner) {
                for(int i=0;i<Nvdf;i++){
                    gradh_t[i*dim+0] += hf[i] * Sf.x;
                    gradh_t[i*dim+1] += hf[i] * Sf.y;
                }
            } else if (cellI == neigh) {
                for(int i=0;i<Nvdf;i++){
                    gradh_t[i*dim+0] -= hf[i] * Sf.x;
                    gradh_t[i*dim+1] -= hf[i] * Sf.y;
                }
            }
        }
        for(int i=0;i<Nvdf;i++){
            int index = index_grad(vi, i, 0);
            gradh[index] = gradh_t[i*dim+0] / V;
            gradh[index+1] = gradh_t[i*dim+1] / V;
        }
    }
}

// =========================
// dJ/dA, dJ/dn
// =========================
void BoundarySensitivityAssembler::accumulate_dJdA_dJdn(
    dvmSolver& primal,
    int facei,
    FaceGeomGrad& g)
{
    const auto& f = primal.mesh.faces[facei];

    int neigh = f.neigh;
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    scalar dJdA = 0.0;
    scalar dJdn[dim] = {Zero};

    if(f.bc_type == BCType::pressure_far_field) {
        for (int vi = 0; vi < primal.Nv; ++vi) {
            double vn = nf.x*primal.Vx[vi] + nf.y*primal.Vy[vi];
            
            int idx_neigh = primal.index_vdf(neigh, vi);
            scalar h1 = primal.vdf[idx_neigh+0];
            double m = objectiveWeight(primal.Vx[vi], primal.Vy[vi]);

            double w = primal.feq[vi] * primal.weight[vi];
    
            dJdA += vn * m * h1 * w;
            dJdn[0] += A * m * h1 * primal.Vx[vi] * w;
            dJdn[1] += A * m * h1 * primal.Vy[vi] * w;
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
    dvmSolver& primal,
    int facei,
    const std::vector<scalar>& gradh,
    FaceGeomGrad& g)
{
    const auto& mesh = primal.mesh;
    const auto& f = mesh.faces[facei];

    const vector& r1 = mesh.points[f.n1];
    const vector& r2 = mesh.points[f.n2];

    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    double dJdC[2] = {0.0, 0.0};

    if(f.bc_type == BCType::pressure_far_field){
        for (int vi = 0; vi < primal.Nv; ++vi) {
            double vn = nf.x*primal.Vx[vi] + nf.y*primal.Vy[vi];
            double m  = objectiveWeight(primal.Vx[vi], primal.Vy[vi]);
    
            double coeff = A*vn*m*primal.feq[vi] * primal.weight[vi];
    
            int idx_grad = index_grad(vi, 0, 0);
            dJdC[0] += coeff * gradh[idx_grad+0];
            dJdC[1] += coeff * gradh[idx_grad+1];
         }
    }

    g.dJdC[0] = dJdC[0];
    g.dJdC[1] = dJdC[1];
}

// =========================
// dBw/dn
// =========================
void BoundarySensitivityAssembler::accumulate_dBwdn(
    dvmSolver& primal,
    const adjointDVM& adjoint,
    int facei,
    FaceGeomGrad& g)
{
    const auto& f = primal.mesh.faces[facei];
    int owner = f.owner;
    int neigh = f.neigh;
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    // compute dK/dn
    double dKdn[2] = {0.0, 0.0};
    for (int vi = 0; vi < primal.Nv; ++vi) {
        double cn = nf.x * primal.Vx[vi] + nf.y * primal.Vy[vi];
        if (cn > 0.0) {
            int idx_neigh = primal.index_vdf(neigh, vi);
            double h1 = primal.vdf[idx_neigh+0];
            double coeff = 2.0*sqrtPI* primal.feq[vi] * h1 * primal.weight[vi];

            dKdn[0] += coeff * primal.Vx[vi];
            dKdn[1] += coeff * primal.Vy[vi];
        }
    }

    double dBwdn[2] = {0.0, 0.0};
    if(f.bc_type == BCType::pressure_far_field) {
        for (int vi = 0; vi < primal.Nv; ++vi) {
            double vn = nf.x*primal.Vx[vi] + nf.y*primal.Vy[vi];
            if (vn >= 0.0) continue;
            
            int idx_neigh = primal.index_vdf(neigh, vi);
            scalar phi1 = adjoint.avdf[idx_neigh+0];
            scalar phi2 = adjoint.avdf[idx_neigh+1];
            double m = objectiveWeight(primal.Vx[vi], primal.Vy[vi]);
            double phiw = -vn * (phi1 + phi2 +  m);
    
            double w = primal.feq[vi] * primal.weight[vi];
            auto dhbdn = DiffuseBoundaryModel::dhb_dn(primal, facei, vi);
            dBwdn[0] -= phiw * (dKdn[0] + dhbdn[0]) * w;
            dBwdn[1] -= phiw * (dKdn[1] + dhbdn[1]) * w;
    
        }
    }else if(f.bc_type == BCType::wall) {
        for (int vi = 0; vi < primal.Nv; ++vi) {
            double vn = nf.x*primal.Vx[vi] + nf.y*primal.Vy[vi];
            if (vn >= 0.0) continue;
    
            int idx_neigh = primal.index_vdf(neigh, vi);
            scalar phi1 = adjoint.avdf[idx_neigh+0];
            scalar phi2 = adjoint.avdf[idx_neigh+1];
            double phiw = -vn * (phi1 + phi2);
    
            double w = primal.feq[vi] * primal.weight[vi];
            auto dhbdn = DiffuseBoundaryModel::dhb_dn(primal, facei, vi);
            dBwdn[0] -= phiw * (dKdn[0] + dhbdn[0]) * w;
            dBwdn[1] -= phiw * (dKdn[1] + dhbdn[1]) * w;
    
        }
    }
    g.dBwdn[0] = A * dBwdn[0];
    g.dBwdn[1] = A * dBwdn[1];
}

// =========================
// dBw/dC
// =========================
void BoundarySensitivityAssembler::accumulate_dBwdC(
    dvmSolver& primal,
    const adjointDVM& adjoint,
    int facei,
    const std::vector<double>& gradh,
    FaceGeomGrad& g)
{
    const auto& f = primal.mesh.faces[facei];
    int owner = f.owner;
    int neigh = f.neigh;
    double A = f.Sf.mag();
    vector nf = f.Sf / A;

    // compute dK/dC
    double dKdC[2] = {0.0, 0.0};
    for (int vi = 0; vi < primal.Nv; ++vi) {
        double vn = nf.x*primal.Vx[vi] + nf.y*primal.Vy[vi];
        if (vn <= 0.0) continue;

        double coeff = 2.0 * sqrtPI* vn * primal.feq[vi] * primal.weight[vi];

        // dh1dx, dh1dy
        int idx_grad = index_grad(vi, 0, 0);
        dKdC[0] += coeff * gradh[idx_grad+0];
        dKdC[1] += coeff * gradh[idx_grad+1];
    }

    // dBw/dC
    double dBwdC[2] = {0.0, 0.0};
    if(f.bc_type == BCType::pressure_far_field) {
        for (int vi = 0; vi < primal.Nv; ++vi) {
            double vn = nf.x*primal.Vx[vi] + nf.y*primal.Vy[vi];
            double w = primal.feq[vi] * primal.weight[vi];
            if (vn >= 0.0) continue;
            
            int idx_neigh = primal.index_vdf(neigh, vi);
            scalar phi1 = adjoint.avdf[idx_neigh+0];
            scalar phi2 = adjoint.avdf[idx_neigh+1];
            double m = objectiveWeight(primal.Vx[vi], primal.Vy[vi]);
            scalar phiw1 = -vn * (phi1 + m);
            scalar phiw2 = -vn * phi2;

            // phiw1*dh1 + phiw2*dh2 - (phiw1+phiw2)*dKdC
            int idx_grad = index_grad(vi, 0, 0);
            dBwdC[0] += w*(phiw1*gradh[idx_grad+0]+phiw2*gradh[idx_grad+2] 
                - (phiw1 + phiw2) * dKdC[0]);
            dBwdC[1] += w*(phiw1*gradh[idx_grad+1]+phiw2*gradh[idx_grad+3] 
                - (phiw1 + phiw2) * dKdC[1]);    
        }
    }else if(f.bc_type == BCType::wall) {
        for (int vi = 0; vi < primal.Nv; ++vi) {
            double vn = nf.x*primal.Vx[vi] + nf.y*primal.Vy[vi];
            double w = primal.feq[vi] * primal.weight[vi];
            if (vn >= 0.0) continue;

            int idx_neigh = primal.index_vdf(neigh, vi);
            scalar phi1 = adjoint.avdf[idx_neigh+0];
            scalar phi2 = adjoint.avdf[idx_neigh+1];
            double m = 0.0;
            scalar phiw1 = -vn * (phi1 + m);
            scalar phiw2 = -vn * phi2;


            // phiw1*dh1 + phiw2*dh2 - (phiw1+phiw2)*dKdC
            int idx_grad = index_grad(vi, 0, 0);
            dBwdC[0] += w*(phiw1*gradh[idx_grad+0]+phiw2*gradh[idx_grad+2] 
                - (phiw1 + phiw2) * dKdC[0]);
            dBwdC[1] += w*(phiw1*gradh[idx_grad+1]+phiw2*gradh[idx_grad+3] 
                - (phiw1 + phiw2) * dKdC[1]);    
        }
    }
    g.dBwdC[0] = A * dBwdC[0];
    g.dBwdC[1] = A * dBwdC[1];
}

// =========================
// face assembly
// =========================
void BoundarySensitivityAssembler::assembleFaceGradients(
    dvmSolver& primal,
    const adjointDVM& adjoint,
    std::vector<FaceGeomGrad>& faceGrad)
{
    const auto& mesh = primal.mesh;
    faceGrad.assign(mesh.nFaces, FaceGeomGrad{});

    std::vector<double> gradh;
    gradh.resize(primal.Nv * dim * Nvdf);

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        if (f.bc_type != BCType::wall &&
            f.bc_type != BCType::pressure_far_field) {
            continue;
        }

        int owner = f.owner;

        FaceGeomGrad g;

        // compute gradient of owner cell
        computeOwnerCellGradient(primal, owner, gradh);

        // objective terms
        accumulate_dJdA_dJdn(primal, facei, g);
        accumulate_dJdC(primal, facei, gradh, g);

        // boundary constraint terms
        accumulate_dBwdn(primal, adjoint, facei, g);
        accumulate_dBwdC(primal, adjoint, facei, gradh, g);

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