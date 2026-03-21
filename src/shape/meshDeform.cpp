#include "meshDeform.h"
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include "math.h"

void buildPointNeighbors(const Mesh& mesh,
                         std::vector<std::vector<int>>& pointNbrs)
{
    const int nPoints = static_cast<int>(mesh.points.size());
    pointNbrs.assign(nPoints, {});

    std::vector<std::unordered_set<int>> tmp(nPoints);

    for (const auto& f : mesh.faces)
    {
        if (f.n1 < 0 || f.n2 < 0) continue;
        tmp[f.n1].insert(f.n2);
        tmp[f.n2].insert(f.n1);
    }

    for (int i = 0; i < nPoints; ++i)
    {
        pointNbrs[i].assign(tmp[i].begin(), tmp[i].end());
    }
}
void collectBoundaryPointsByBC(const Mesh& mesh, BCType targetBC,
    std::vector<int>& pointIds)
{
    std::unordered_set<int> pset;
    for (const auto& f : mesh.faces)
    {
        if (f.neigh == -1 && f.bc_type == targetBC)
    {
        if (f.n1 >= 0) pset.insert(f.n1);
        if (f.n2 >= 0) pset.insert(f.n2);
    }
    }
    pointIds.assign(pset.begin(), pset.end());
}

void collectAllBoundaryPoints(const Mesh& mesh, std::vector<int>& pointIds)
{
    std::unordered_set<int> pset;

    for (const auto& f : mesh.faces)
    {
        if (f.neigh == -1)
        {
            if (f.n1 >= 0) pset.insert(f.n1);
            if (f.n2 >= 0) pset.insert(f.n2);
        }
    }

    pointIds.assign(pset.begin(), pset.end());
}
bool springSmooth(Mesh& mesh,
    const std::vector<BoundaryNodeDisplacement>& bdisp,
    std::vector<vector>& disp,
    int maxIter,
    double tol,
    double omega)
{
    const int nPoints = static_cast<int>(mesh.points.size());   

    std::vector<std::vector<int>> pointNbrs;
    buildPointNeighbors(mesh, pointNbrs);

    std::vector<char> isFixed(mesh.points.size(),0);
    disp.assign(nPoints, vector(0.0, 0.0, 0.0));

    for(auto& bp : bdisp){
        disp[bp.pointId] = bp.disp;
        isFixed[bp.pointId] = 1;
    }

    for (int iter = 0; iter < maxIter; ++iter)
    {
        double maxMove = 0.0;

        for (int i = 0; i < nPoints; ++i)
        {
            if (isFixed[i]) continue; // 边界点固定

            const auto& nbrs = pointNbrs[i];
            if (nbrs.empty()) continue;

            double wsum = 0.0;
            double sum[3]={0.0};

            const vector& pi = mesh.points[i];

            for (int j : nbrs)
            {
                const vector& pj = mesh.points[j];
                double L = (pi-pj).mag();
                double k = 1.0 / L;

                sum[0] += k*(disp[j].x-disp[i].x);
                sum[1] += k*(disp[j].y-disp[i].y);
                sum[2] += k*(disp[j].z-disp[i].z);
                wsum += k;
            }

            if (wsum < 1e-14) continue;

            double ddx = omega * sum[0] / wsum;
            double ddy = omega * sum[1] / wsum;
            double ddz = omega * sum[2] / wsum;
            double move = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);

            disp[i].x += ddx;
            disp[i].y += ddy;
            disp[i].z += ddz;

            if (move > maxMove) maxMove = move;
        }

        if (maxMove < tol)
        {
            printf("[springSmooth] converged at iter = %d, maxMove = %.3e\n",iter,maxMove);
            return true;
        }
    }
    printf("[springSmooth] reached maxIter without full convergence.\n");
    return false;
}

static void updateLocalMeshPoints(
    Mesh& localMesh,
    const std::vector<double>& globalPoints)
{
    const int nLocalPoints = static_cast<int>(localMesh.points.size());

    if ((int)localMesh.l2g_point.size() != nLocalPoints)
    {
        printf("mesh.l2g_point.size() != mesh.points.size()");
    }

    for (int lp = 0; lp < nLocalPoints; ++lp)
    {
        int gp = localMesh.l2g_point[lp];
        localMesh.points[lp].x = globalPoints[3 * gp + 0];
        localMesh.points[lp].y = globalPoints[3 * gp + 1];
        localMesh.points[lp].z = globalPoints[3 * gp + 2];
    }
}
// TODO: 后面可以添加PDE广顺的方法（适合大变形），提供多种选择
void deformMeshSpring(Mesh& globalMesh,
    Mesh& localMesh,
    std::vector<BoundaryNodeDisplacement>& bdisp,
    MPI_Comm comm,
    int maxIter,
    double tol,
    double omega)
{
    int rank;
    MPI_Comm_rank(comm,&rank);

    std::vector<double> globalPoints;
    int nGlobalPoints=0;

    if(rank==0){
        // compute displacement for nodes
        std::vector<vector> disp;
        springSmooth(globalMesh, bdisp, disp, maxIter, tol, omega);
        // add to globalMesh
        nGlobalPoints = globalMesh.points.size();
        globalPoints.resize(3*nGlobalPoints);
        for (int i = 0; i < (int)globalMesh.points.size(); ++i)
        {
            globalMesh.points[i].x += disp[i].x;
            globalMesh.points[i].y += disp[i].y;
            globalMesh.points[i].z += disp[i].z;

            globalPoints[i*3+0] = globalMesh.points[i].x;
            globalPoints[i*3+1] = globalMesh.points[i].y;
            globalPoints[i*3+2] = globalMesh.points[i].z;
        }
    }
    // cast nGlobalPoints to all ranks
    MPI_Bcast(&nGlobalPoints, 1, MPI_INT, 0, comm);
    if(rank!=0){
        globalPoints.resize(3*nGlobalPoints);
    }
    // cast to all ranks
    MPI_Bcast(globalPoints.data(), 3 * nGlobalPoints, MPI_DOUBLE, 0, comm);
    // update local points
    updateLocalMeshPoints(localMesh,globalPoints);
    // compute geometry
    computeGeometry(localMesh, comm);// check mesh
}