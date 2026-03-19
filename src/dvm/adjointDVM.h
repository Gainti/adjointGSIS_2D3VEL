#ifndef ADJOINT_DVM_H
#define ADJOINT_DVM_H

#include "dvmSolver.h"
#include <vector>

class adjointDVM {
public:
    dvmSolver& primal;

    const Mesh& mesh;
    const SolverConfig& cfg;
    MPI_Comm comm;
    int rank;
    int size;
    const int Nv;

    const std::vector<double>& Vx;
    const std::vector<double>& Vy;
    const std::vector<double>& c2;
    const std::vector<double>& feq;
    const std::vector<double>& weight;
    const std::vector<double>& invdt;

    std::vector<scalar> avdf, arhs;
    std::vector<scalar> amacro;

    double res_aux, res_auy, res_arho, res_atau;

    explicit adjointDVM(dvmSolver& primal);

    inline int index_vdf(int celli, int vi) const {
        return primal.index_vdf(celli, vi);
    }
    
    void updateAdjMacro();
    void adjointBoundarySet();
    void getAdjRhs();
    void cellIterAdj(int cellI);
    void lusgsIterAdj();
    void step(int iter);
};

#endif // ADJOINT_DVM_H
