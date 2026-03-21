#ifndef ADJOINT_DVM_H
#define ADJOINT_DVM_H

#include "dvmSolver.h"
#include <vector>
#include "object.h"

class adjointDVM {
public:
    // config
    const SolverConfig& cfg;
    // MPI
    MPI_Comm comm;
    int rank;
    int size;
    // mesh
    const Mesh& mesh;
    const int Nv;
    // velocity space
    const std::vector<double>& Vx;
    const std::vector<double>& Vy;
    const std::vector<double>& v2;
    const std::vector<double>& feq;
    const std::vector<double>& weight;
    const std::vector<std::array<double, Nmacro*2>> &weight_macro,&weight_coll;
    // adjoint vdf
    std::vector<scalar> avdf, arhs;
    // macro
    std::vector<scalar> amacro;
    // pseudo time step
    std::vector<double> invdt;

    double res_aux, res_auy, res_arho, res_atau;

    explicit adjointDVM(const Mesh& mesh,
        const VelocitySpace& vel,
        const SolverConfig& cfg,
        MPI_Comm comm);

    inline int index_vdf(int celli, int vi) const {
        return celli*Nv*Nvdf+vi*Nvdf;
    }
    
    void massConservation();
    void updateAdjMacro();
    void adjointBoundarySet();
    void getAdjRhs();
    void cellIterAdj(int cellI);
    void lusgsIterAdj();
    void step(int iter);

    // void grad(int cellI,int vi, double& gradx, double& grady);
    void grad(int cellI,int vi, scalar* gradh,int nvdf);

private:
    void diffuseWall(int facei);
    void diffuseWallwithObject(int facei);
};

#endif // ADJOINT_DVM_H
