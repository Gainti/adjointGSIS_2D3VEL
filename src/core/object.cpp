#include "object.h"


double objectiveWeight(double vx, double vy){
    return 2.0*vx;
}

// 计算pressure_far_field边界上的目标函数值
double computeObjective(dvmSolver& solver)
{
    const auto& mesh = solver.mesh;
    double J_local = 0.0;

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; ++facei) {
        const auto& f = mesh.faces[facei];
        if (f.bc_type != BCType::pressure_far_field) {
            continue;
        }

        double A = f.Sf.mag();
        vector nf = f.Sf / A;
        int ghost = f.neigh;

        for (int vi = 0; vi < solver.Nv; ++vi) {
            double cn = nf.x * solver.Vx[vi] + nf.y * solver.Vy[vi];
            int idx_g = solver.index_vdf(ghost, vi);

            double h = solver.vdf[idx_g];
            double m = objectiveWeight(solver.Vx[vi], solver.Vy[vi]);

            J_local += A * cn * m * h * solver.feq[vi] * solver.weight[vi];
        }
    }

    double J_global = 0.0;
    MPI_Allreduce(&J_local, &J_global, 1, MPI_DOUBLE, MPI_SUM, solver.comm);
    return J_global;
}