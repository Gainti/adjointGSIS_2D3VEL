#ifndef ADJOINT_VALIDATION_H
#define ADJOINT_VALIDATION_H

#include "mesh.h"
#include "dvmSolver.h"
#include "defs.h"
#include "mpi.h"

#include <vector>
#include <string>

#include "velocitySpace.h"
#include "object.h"
#include "boundary_sensitivity.h"

#include "meshDeform.h"

bool isBoundaryFace(const Face& f);
double localReferenceLength(const Mesh& mesh, int point_id);

// 跑一次 primal，返回目标函数 J
double runPrimalAndEvalJ(const Mesh& mesh,
    const VelocitySpace& vel,
    const SolverConfig& cfg,
    MPI_Comm comm);

// 只动一个点的一个坐标分量
void perturbOnePoint(Mesh& mesh, int point_id, int coord, double ds);

// 有限差分验证主入口
bool validateOneBoundaryPoint(const Mesh& base_mesh,
    const VelocitySpace& vel,
    const SolverConfig& cfg,
    MPI_Comm comm);


// 计算关于scale的伴随导数
double projectGradientToYScalingDirection(
    const Mesh& mesh,
    const std::vector<NodeGrad>& nodeGrad,
    MPI_Comm comm);

// 将网格整体沿 y 方向缩放，验证伴随导数在这个方向上的投影
bool validateYscaleboundary(const Mesh& globalMesh,
    const Mesh& localMesh,
    const VelocitySpace& vel,
    const SolverConfig& cfg,
    MPI_Comm comm);

#endif