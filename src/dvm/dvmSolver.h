#ifndef DVMSOLVER_H
#define DVMSOLVER_H

#include "defs.h"
#include <array>
#include <vector>
#include <fstream>
#include "mesh.h"
#include "mpi.h"
#include "halo.h"
#include "profiler.h"
#include "velocitySpace.h"

class dvmSolver {
public:
    // config
    const SolverConfig& cfg;
    // MPI
    MPI_Comm comm;
    int rank,size;
    DvmProfiler profiler;
    // mesh
    const Mesh &mesh;
    const int Nv;
    const std::vector<double> &Vx,&Vy;
    const std::vector<double>& weight;
    const std::vector<double>& v2;
    const std::vector<double>& feq;
    const std::vector<std::array<double, Nmacro*2>> &weight_macro,&weight_coll;
    // vdf
    std::vector<scalar> vdf,rhs;
    HaloWorkspace halo_ws;
    // macro
    std::vector<scalar> macro;
    std::vector<scalar> stress,hot_stress;// dim*dim
    std::vector<scalar> hot_heat;// dim
    // beta
    std::vector<double> beta;
    // 1/dt
    std::vector<double> invdt;

    // output
    double res_ux,res_uy,res_rho,res_tau;

    // 构造函数
    dvmSolver(const Mesh& mesh,
        const VelocitySpace& vel,
        const SolverConfig& cfg,
        MPI_Comm comm);
    ~dvmSolver(){}

    inline int index_vdf(int celli,int vi){
        return celli*Nv*Nvdf+vi*Nvdf;
    }

    // primal
    void step(int iter);
    void calHot();
    void boundarySet();
    void updateMacro();
    void massConservation();
    void getRhs();
    void lusgsIter();
    void cellIter(int cellI);
    void sweepCells(const std::vector<int>& cellList, bool forward);
    void grad(int cellI,int vi, scalar* gradh,int nvdf);

    void reportProfile() const {
        profiler.report(comm, rank, "[DVM profile summary]");
    }

private:
    void diffuseWall(int facei, double uwx, double uwy);
};
#endif //DVMSOLVER_H
