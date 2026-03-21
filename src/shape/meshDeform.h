#pragma once

#include "mesh.h"
#include "mpi.h"
#include <functional>
#include "computeGeometry.h"

// 给定边界点位移
struct BoundaryNodeDisplacement
{
    int pointId;   // local point id
    vector disp;   // 位移
};

// 构建点邻接关系（弹簧边）
void buildPointNeighbors(const Mesh& mesh,
    std::vector<std::vector<int>>& pointNbrs);

static void updateLocalMeshPoints(
    Mesh& localMesh,
    const std::vector<double>& globalPoints);

// 弹簧光顺：更新内部节点
bool springSmooth(Mesh& mesh,
    const std::vector<BoundaryNodeDisplacement>& bdisp,
    std::vector<vector>& disp,
    int maxIter = 200,
    double tol = 1e-10,
    double omega = 1.0);

// 网格变形总入口
void deformMeshSpring(Mesh& globalMesh,
    Mesh& localMesh,
    std::vector<BoundaryNodeDisplacement>& bdisp,
    MPI_Comm comm,
    int maxIter = 200,
    double tol = 1e-10,
    double omega = 1.0);

// 工具函数：收集某一类边界上的点
void collectBoundaryPointsByBC(const Mesh& mesh,
    BCType targetBC,
    std::vector<int>& pointIds);

// 工具函数：收集全部边界点
void collectAllBoundaryPoints(const Mesh& mesh,
    std::vector<int>& pointIds);