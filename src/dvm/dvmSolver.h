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
    MPI_Comm comm;
    int rank,size;

    DvmProfiler profiler;
    // mesh
    const Mesh &mesh;
    const int Nv;

    HaloWorkspace halo_ws;

    // constant
    SolverConfig cfg;

    const std::vector<double> &Vx,&Vy,&Vz;
    const std::vector<double>& weight;
    const std::vector<double>& c2;
    const std::vector<double>& feq;
    const std::vector<double>& exp_c2;
    const std::vector<std::array<double, Nmacro>>& weight_macro;
    // vdf
    std::vector<scalar> vdf,rhs;
    // macro
    std::vector<scalar> macro;
    std::vector<scalar> hot;
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
        return celli*Nv+vi;
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
    void grad(int cellI,int vi, double& gradx, double& grady);

    void reportProfile() const {
        profiler.report(comm, rank, "[DVM profile summary]");
    }

private:
    void diffuseWall(int facei, double uwx, double uwy);
};
#endif //DVMSOLVER_H
