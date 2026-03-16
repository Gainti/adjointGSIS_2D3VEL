#include "geom_chain_validation.h"

#include <cmath>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

// ----------------------------
// local helpers
// ----------------------------
namespace
{
    inline bool isBoundaryFaceForShape(const Face& f)
    {
        return (f.bc_type == BCType::wall ||
                f.bc_type == BCType::pressure_far_field);
    }

    inline std::array<double, 2> make2(double x, double y)
    {
        return {x, y};
    }

    inline double relErr(double a, double b)
    {
        double denom = std::max({1.0, std::fabs(a), std::fabs(b)});
        return std::fabs(a - b) / denom;
    }

    inline void perturbPoint(Mesh& mesh, int nodeId, int comp, double eps)
    {
        if (comp == 0) mesh.points[nodeId].x += eps;
        else           mesh.points[nodeId].y += eps;
    }

    inline std::array<double,2> edgeNormal(const vector& r1, const vector& r2)
    {
        double dx = r2.x - r1.x;
        double dy = r2.y - r1.y;
        double A  = std::sqrt(dx * dx + dy * dy);
        if (A <= 1e-30)
        {
            throw std::runtime_error("degenerate edge in edgeNormal()");
        }
        return {dy / A, -dx / A};
    }

    inline double edgeLength(const vector& r1, const vector& r2)
    {
        double dx = r2.x - r1.x;
        double dy = r2.y - r1.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    inline std::array<double,2> edgeCenter(const vector& r1, const vector& r2)
    {
        return {0.5 * (r1.x + r2.x), 0.5 * (r1.y + r2.y)};
    }
}

// ----------------------------
// frozen geometric functional
// ----------------------------
double evalFrozenGeomFunctional(
    const Mesh& mesh,
    const std::vector<FaceGeomGrad>& faceGrad)
{
    double L = 0.0;

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei)
    {
        const auto& f = mesh.faces[facei];
        if (!isBoundaryFaceForShape(f)) continue;

        const vector& r1 = mesh.points[f.n1];
        const vector& r2 = mesh.points[f.n2];

        double A = edgeLength(r1, r2);
        auto C   = edgeCenter(r1, r2);
        auto n   = edgeNormal(r1, r2);

        const auto& g = faceGrad[facei];

        L += g.dLdA    * A;
        L += g.dLdC[0] * C[0] + g.dLdC[1] * C[1];
        L += g.dLdn[0] * n[0] + g.dLdn[1] * n[1];
    }

    return L;
}

// ----------------------------
// level-0: primitive FD check
// ----------------------------
void validateEdgeGeomPrimitiveFD(
    const Mesh& mesh,
    double eps)
{
    std::printf("\n========== [Level-0] validateEdgeGeomPrimitiveFD ==========\n");

    double maxErr_dA = 0.0;
    double maxErr_dC = 0.0;
    double maxErr_dn = 0.0;
    int    nChecked  = 0;

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei)
    {
        const auto& f = mesh.faces[facei];
        if (!isBoundaryFaceForShape(f)) continue;

        const vector& r1 = mesh.points[f.n1];
        const vector& r2 = mesh.points[f.n2];

        EdgeGeomDeriv2D gd = computeEdgeGeomDeriv2D(r1, r2);

        double A0 = gd.A;
        auto   C0 = make2(gd.C.x, gd.C.y);
        auto   n0 = make2(gd.n.x, gd.n.y);

        for (int s = 0; s < 2; ++s)
        {
            for (int comp = 0; comp < 2; ++comp)
            {
                vector rr1 = r1;
                vector rr2 = r2;

                if (s == 0)
                {
                    if (comp == 0) rr1.x += eps;
                    else           rr1.y += eps;
                }
                else
                {
                    if (comp == 0) rr2.x += eps;
                    else           rr2.y += eps;
                }

                double Ap = edgeLength(rr1, rr2);
                auto   Cp = edgeCenter(rr1, rr2);
                auto   np = edgeNormal(rr1, rr2);

                double fd_dA = (Ap - A0) / eps;
                double an_dA = (s == 0 ? gd.dA_dr1[comp] : gd.dA_dr2[comp]);
                maxErr_dA = std::max(maxErr_dA, relErr(fd_dA, an_dA));

                for (int j = 0; j < 2; ++j)
                {
                    double fd_dC = (Cp[j] - C0[j]) / eps;
                    double an_dC = gd.dCdr[s][j][comp];
                    maxErr_dC = std::max(maxErr_dC, relErr(fd_dC, an_dC));
                }

                for (int j = 0; j < 2; ++j)
                {
                    double fd_dn = (np[j] - n0[j]) / eps;
                    double an_dn = gd.dndr[s][j][comp];
                    maxErr_dn = std::max(maxErr_dn, relErr(fd_dn, an_dn));
                }

                ++nChecked;
            }
        }
    }

    std::printf("checked primitive perturbations = %d\n", nChecked);
    std::printf("max relative error of dA/dr     = %.6e\n", maxErr_dA);
    std::printf("max relative error of dC/dr     = %.6e\n", maxErr_dC);
    std::printf("max relative error of dn/dr     = %.6e\n", maxErr_dn);
}

// ----------------------------
// level-1: chain-rule check
// ----------------------------
void validateNodeChainFD(
    const Mesh& mesh,
    const std::vector<FaceGeomGrad>& faceGrad,
    double eps)
{
    std::printf("\n========== [Level-1] validateNodeChainFD ==========\n");

    std::vector<NodeGrad> nodeGrad;
    BoundarySensitivityAssembler::accumulateNodeGradients(mesh, faceGrad, nodeGrad);

    std::set<int> boundaryNodes;
    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei)
    {
        const auto& f = mesh.faces[facei];
        if (!isBoundaryFaceForShape(f)) continue;
        boundaryNodes.insert(f.n1);
        boundaryNodes.insert(f.n2);
    }

    const double L0 = evalFrozenGeomFunctional(mesh, faceGrad);

    double maxErrX = 0.0;
    double maxErrY = 0.0;

    std::printf("number of boundary nodes checked = %zu\n", boundaryNodes.size());

    for (int nodeId : boundaryNodes)
    {
        {
            Mesh mp = mesh;
            perturbPoint(mp, nodeId, 0, eps);
            double Lp = evalFrozenGeomFunctional(mp, faceGrad);
            double fd = (Lp - L0) / eps;
            double an = nodeGrad[nodeId].dx;
            double er = relErr(fd, an);
            maxErrX = std::max(maxErrX, er);

            std::printf("[node %d, x] FD = %.6e, AN = %.6e, relErr = %.6e\n",
                        nodeId, fd, an, er);
        }

        {
            Mesh mp = mesh;
            perturbPoint(mp, nodeId, 1, eps);
            double Lp = evalFrozenGeomFunctional(mp, faceGrad);
            double fd = (Lp - L0) / eps;
            double an = nodeGrad[nodeId].dy;
            double er = relErr(fd, an);
            maxErrY = std::max(maxErrY, er);

            std::printf("[node %d, y] FD = %.6e, AN = %.6e, relErr = %.6e\n",
                        nodeId, fd, an, er);
        }
    }

    std::printf("max relative error in dL/dx = %.6e\n", maxErrX);
    std::printf("max relative error in dL/dy = %.6e\n", maxErrY);
}

// ----------------------------
// public entry
// ----------------------------
void runGeometryChainValidation(
    Mesh& mesh,
    double eps)
{
    std::printf("\n====================================================\n");
    std::printf("Run geometry-chain validation (no prime terms)\n");
    std::printf("eps = %.6e\n", eps);
    std::printf("====================================================\n");

    std::vector<FaceGeomGrad> faceGrad(mesh.faces.size());

    for (auto& g : faceGrad)
    {
        g.dLdA    = 1.0;
        g.dLdC[0] = 1.0;
        g.dLdC[1] = 1.0;
        g.dLdn[0] = 1.0;
        g.dLdn[1] = 1.0;
    }

    validateEdgeGeomPrimitiveFD(mesh, eps);
    validateNodeChainFD(mesh, faceGrad, eps);

    std::printf("====================================================\n");
}