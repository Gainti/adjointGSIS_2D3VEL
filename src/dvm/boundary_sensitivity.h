#ifndef BOUNDARY_SENSITIVITY_H
#define BOUNDARY_SENSITIVITY_H

#include "mesh.h"
#include "dvmSolver.h"
#include <array>
#include <vector>

struct FaceGeomGrad
{
    double dLdA = 0.0;
    std::array<double,2> dLdC{0.0, 0.0};
    std::array<double,2> dLdn{0.0, 0.0};
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
// 目标函数接口
// =========================
class BoundaryFunctional
{
public:
    virtual ~BoundaryFunctional() = default;
    virtual double mPlus(const dvmSolver& solver, int facei, int vi) const = 0;
    virtual double mMinus(const dvmSolver& solver, int facei, int vi) const = 0;
};

class XForceFunctional : public BoundaryFunctional
{
public:
    double mPlus(const dvmSolver& solver, int facei, int vi) const override;
    double mMinus(const dvmSolver& solver, int facei, int vi) const override;
};

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
        dvmSolver& solver, int facei, int owner);

    static std::array<double,2> wallVelocity(
        dvmSolver& solver, int facei);

    static double hb(
        dvmSolver& solver, int facei, int vi);

    static std::array<double,2> dhb_dn(
        dvmSolver& solver, int facei, int vi);
};

// =========================
// 灵敏度装配
// =========================
class BoundarySensitivityAssembler
{
public:
    static void assembleFaceGradients(
        dvmSolver& solver,
        const BoundaryFunctional& obj,
        std::vector<FaceGeomGrad>& faceGrad);

    static void accumulateNodeGradients(
        const Mesh& mesh,
        const std::vector<FaceGeomGrad>& faceGrad,
        std::vector<NodeGrad>& nodeGrad);

private:
    // 只为当前 owner 单元计算所有速度点梯度
    static void computeOwnerCellGradient(
        dvmSolver& solver,
        int owner,
        std::vector<double>& gradHx,
        std::vector<double>& gradHy);

    static void accumulate_dJdA_dJdn(
        dvmSolver& solver,
        int facei,
        const BoundaryFunctional& obj,
        FaceGeomGrad& g);

    static void accumulate_dJdC(
        dvmSolver& solver,
        int facei,
        const BoundaryFunctional& obj,
        const std::vector<double>& gradHx,
        const std::vector<double>& gradHy,
        FaceGeomGrad& g);

    static std::array<double,2> compute_dKdC(
        dvmSolver& solver,
        int facei,
        int owner,
        const std::vector<double>& gradHx,
        const std::vector<double>& gradHy);

    static void accumulate_dBwdn(
        dvmSolver& solver,
        int facei,
        int owner,
        const BoundaryFunctional& obj,
        FaceGeomGrad& g);

    static void accumulate_dBwdC(
        dvmSolver& solver,
        int facei,
        int owner,
        const BoundaryFunctional& obj,
        const std::vector<double>& gradHx,
        const std::vector<double>& gradHy,
        const std::array<double,2>& dKdC,
        FaceGeomGrad& g);
};

#endif