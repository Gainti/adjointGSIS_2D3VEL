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

const int dim=2;
const int Nmacro=10;// rho,ux,uy,tau,qx,qy,pxx,pxy,pyx,pyy
const int Nhot=6;// stress_{xx,xy,yx,yy} + heat_{x,y}

const int Namacro=6;

void uniform(std::vector<double> &vi,
    std::vector<double> &weight_i,size_t Nvi,double Lvi);

void nonUniform(std::vector<double> &vi,std::vector<double> &weight_i,size_t Nvi,double Lvi);

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

    // velocity grid
    std::vector<double> vx,vy,vz;
    std::vector<double> weight_x,weight_y,weight_z;
    std::vector<double> Vx,Vy,Vz;
    std::vector<double> weight;
    std::vector<double> c2,feq,exp_c2;
    std::vector<std::array<double, Nmacro>> weight_macro;
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
    double cantileverOut[dim];

    // ===========adjoint ============
    // adjoint vdf
    std::vector<scalar> avdf, arhs;
    // adjoint macro: rho, ux, uy, tau, qx, qy
    std::vector<scalar> amacro;
    double res_aux,res_auy,res_arho,res_atau;


    // 构造函数
    dvmSolver(const Mesh& mesh,
        const SolverConfig& cfg,
        MPI_Comm comm);
    ~dvmSolver(){}

    inline int index_vdf(int celli,int vi){
        return celli*Nv+vi;
    }

    // primal
    void initial();
    void step(int iter);
    void calHot();
    bool allocateVelSpace();
    void boundarySet();
    void updateMacro();
    void massConservation();
    void getRhs();
    void lusgsIter();
    void cellIter(int cellI);
    void sweepCells(const std::vector<int>& cellList, bool forward);
    
    // adjoint solver
    void initialAdj();
    void updateAdjMacro();
    void adjointBoundarySet();
    void getAdjRhs();
    void cellIterAdj(int cellI);
    void lusgsIterAdj();
    void stepAdj(int iter);


    void reportProfile() const {
        profiler.report(comm, rank, "[DVM profile summary]");
    }
};
#endif //DVMSOLVER_H
