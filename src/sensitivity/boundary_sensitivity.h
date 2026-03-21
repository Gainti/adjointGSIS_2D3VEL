#ifndef BOUNDARY_SENSITIVITY_H
#define BOUNDARY_SENSITIVITY_H

#include "mesh.h"
#include "dvmSolver.h"
#include "adjointDVM.h"
#include <array>
#include <vector>
#include "object.h"

struct FaceGeomGrad
{
    double dJdA;
    double dJdC[2], dBwdC[2];
    double dJdn[2], dBwdn[2];

    double dLdA;
    double dLdC[2];
    double dLdn[2];
};

struct NodeGrad
{
    double dx = 0.0;
    double dy = 0.0;
};

struct EdgeGeomDeriv2D
{
    double A = 0.0;
    vector C;
    vector n;

    std::array<double,2> dA_dr1{0.0, 0.0};
    std::array<double,2> dA_dr2{0.0, 0.0};

    // dC_i / dr_s_j
    double dCdr[2][2][2] = {{{0.0}}};

    // dn_i / dr_s_j
    double dndr[2][2][2] = {{{0.0}}};
};

EdgeGeomDeriv2D computeEdgeGeomDeriv2D(const vector& r1, const vector& r2);

// =========================
// 边界模型
// =========================
class DiffuseBoundaryModel
{
public:
    struct KernelGrad
    {
        double K = 0.0;
        std::array<double,2> dKdn{0.0, 0.0};
    };

    static KernelGrad computeKernelAndGradN(
        dvmSolver& primal, int facei, int owner);

    static std::array<double,2> wallVelocity(
        dvmSolver& primal, int facei);

    static double hb(
        dvmSolver& primal, int facei, int vi);

    static std::array<double,2> dhb_dn(
        dvmSolver& primal, int facei, int vi);
};

// =========================
// 灵敏度装配
// =========================
class BoundarySensitivityAssembler
{
public:
    static void assembleFaceGradients(
        dvmSolver& primal,
        const adjointDVM& adjoint,
        std::vector<FaceGeomGrad>& faceGrad);

    static void accumulateNodeGradients(
        const Mesh& mesh,
        const std::vector<FaceGeomGrad>& faceGrad,
        std::vector<NodeGrad>& nodeGrad);

private:
    // 只为当前 owner 单元计算所有速度点梯度
    static void computeOwnerCellGradient(
        dvmSolver& primal,
        int owner,
        std::vector<double>& gradh);

    static void accumulate_dJdA_dJdn(
        dvmSolver& primal,
        int facei,
        FaceGeomGrad& g);

    static void accumulate_dJdC(
        dvmSolver& primal,
        int facei,
        const std::vector<double>& gradh,
        FaceGeomGrad& g);

    static void accumulate_dBwdn(
        dvmSolver& primal,
        const adjointDVM& adjoint,
        int facei,
        FaceGeomGrad& g);

    static void accumulate_dBwdC(
        dvmSolver& primal,
        const adjointDVM& adjoint,
        int facei,
        const std::vector<double>& gradh,
        FaceGeomGrad& g);
};

#endif