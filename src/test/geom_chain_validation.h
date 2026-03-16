#ifndef GEOM_CHAIN_VALIDATION_H_INCLUDED
#define GEOM_CHAIN_VALIDATION_H_INCLUDED

#include <vector>
#include "boundary_sensitivity.h"

// 只验证几何链条：冻结 faceGrad，不重新装配
double evalFrozenGeomFunctional(
    const Mesh& mesh,
    const std::vector<FaceGeomGrad>& faceGrad);

// 逐条验证单条边的 A/C/n 导数
void validateEdgeGeomPrimitiveFD(
    const Mesh& mesh,
    double eps);

// 验证从 faceGrad -> nodeGrad 的链式回传
void validateNodeChainFD(
    const Mesh& mesh,
    const std::vector<FaceGeomGrad>& faceGrad,
    double eps);

// 几何测试接口：分别进行几何链条和链式回传验证
void runGeometryChainValidation(
    Mesh& mesh,
    double eps);

#endif