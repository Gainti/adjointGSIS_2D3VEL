#ifndef ADJOINT_VALIDATION_H
#define ADJOINT_VALIDATION_H

#include "mesh.h"
#include "dvmSolver.h"
#include "defs.h"
#include "mpi.h"

#include <vector>
#include <string>

#include "boundary_sensitivity.h"
#include "object.hpp"

bool isBoundaryFace(const Face& f);
std::vector<int> collectBoundaryPoints(const Mesh& mesh);
std::vector<int> collectOwnerCellsAroundPoint(const Mesh& mesh, int point_id);
double localReferenceLength(const Mesh& mesh, int point_id);

bool recomputeMeshGeometry(Mesh& mesh);

// 跑一次 primal，返回目标函数 J
double runPrimalAndEvalJ(const Mesh& mesh,
                         const SolverConfig& cfg,
                         MPI_Comm comm,
                         int point_id,
                         int macro_comp);

// 只动一个点的一个坐标分量
void perturbOnePoint(Mesh& mesh, int point_id, int coord, double ds);

// 有限差分验证主入口
bool validateOneBoundaryPoint(const Mesh& base_mesh,
                    const SolverConfig& cfg,
                    MPI_Comm comm);

// 如果你后续有“每个边界面的几何灵敏度”，可用这个把面梯度回收到点梯度
struct FaceGeomGrad_t
{
    double dJ_dCfx = 0.0;
    double dJ_dCfy = 0.0;
    double dJ_dSfx = 0.0;
    double dJ_dSfy = 0.0;
};

double contractFaceGeomGradToPoint(const Mesh& mesh,
                                   const std::vector<FaceGeomGrad_t>& face_grad,
                                   int point_id,
                                   int coord);

#endif