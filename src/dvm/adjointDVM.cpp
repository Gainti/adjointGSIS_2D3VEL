#include "adjointDVM.h"

#include <algorithm>
#include <cmath>

adjointDVM::adjointDVM(dvmSolver& primal)
    : primal(primal),
      mesh(primal.mesh),
      cfg(primal.cfg),
      comm(primal.comm),
      rank(primal.rank),
      size(primal.size),
      Nv(primal.Nv),
      Vx(primal.Vx),
      Vy(primal.Vy),
      c2(primal.c2),
      feq(primal.feq),
      weight(primal.weight),
      invdt(primal.invdt),
      res_aux(0.0),
      res_auy(0.0),
      res_arho(0.0),
      res_atau(0.0)
{
    avdf.resize(Nv * (mesh.nCells + mesh.nBoundaryFaces), Zero);
    arhs.resize(Nv * (mesh.nCells + mesh.nBoundaryFaces), Zero);
    amacro.resize((mesh.nCells + mesh.nBoundaryFaces) * Namacro, Zero);

    adjointBoundarySet();
    updateAdjMacro();
}

void adjointDVM::updateAdjMacro() {
    double up_local[4]={0.0},down_local[4]={0.0};
    for(size_t cellI=0; cellI<mesh.nCells+mesh.nBoundaryFaces; ++cellI) {
        scalar a[6] = {Zero};

        for(size_t vi=0; vi<Nv; ++vi) {
            int idx = index_vdf(cellI, vi);
            scalar phi1 = avdf[idx];

            a[0] += phi1 * feq[vi] * weight[vi];
            a[1] += 2.0 * Vx[vi] * phi1 * feq[vi] * weight[vi];
            a[2] += 2.0 * Vy[vi] * phi1 * feq[vi] * weight[vi];
            a[3] += (c2[vi]-1.5)*phi1 * feq[vi] * weight[vi];
            a[4] += (4.0/15.0) * (c2[vi]-2.5)*phi1 * Vx[vi] * feq[vi] * weight[vi];
            a[5] += (4.0/15.0) * (c2[vi]-2.5)*phi1 * Vy[vi] * feq[vi] * weight[vi];
        }
        if(cellI<mesh.nCells){
            double V=mesh.cells[cellI].V;
            for(int i=0;i<4;i++){
                scalar amacro_old = amacro[cellI*Namacro+i];
                up_local[i]+=(amacro_old-a[i])*(amacro_old-a[i])*V;
                down_local[i]+=a[i]*a[i]*V;
            }
        }
        for(int k=0;k<Namacro;k++) {
            amacro[cellI*Namacro+k] = a[k];
        }
    }
    double up_global[4]={0.0}, down_global[4]={0.0};
    MPI_Allreduce(up_local,   up_global,   4, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(down_local, down_global, 4, MPI_DOUBLE, MPI_SUM, comm);
    const double eps = 1e-300;
    res_arho = std::sqrt(up_global[0] / std::max(down_global[0], eps));
    res_aux  = std::sqrt(up_global[1] / std::max(down_global[1], eps));
    res_auy  = std::sqrt(up_global[2] / std::max(down_global[2], eps));
    res_atau = std::sqrt(up_global[3] / std::max(down_global[3], eps));
}
// TODO: 目前定义m为2*Vx
void adjointDVM::adjointBoundarySet() {
    const auto& faces = mesh.faces;
    for(int facei=mesh.nInternalFaces;facei<mesh.nFaces;facei++){
        if(faces[facei].bc_type==BCType::wall ||
        faces[facei].bc_type==BCType::pressure_far_field){
            int owner=faces[facei].owner;
            int neigh=faces[facei].neigh;
            const vector& Sf=faces[facei].Sf;
            double magSf=Sf.mag();
            vector nf= Sf/magSf;
            
            // 积分获得rhor
            scalar rhor=Zero;
            for(int vi=0;vi<Nv;vi++) {
                double cn=nf.x*Vx[vi]+nf.y*Vy[vi];
                if (cn<0.0) {
                    int idx_owner = index_vdf(owner,vi);
                    int idx_neigh = index_vdf(neigh,vi);
                    avdf[idx_neigh] = avdf[idx_owner];
                    double m=0.0;
                    if(faces[facei].bc_type==BCType::pressure_far_field){
                        m=2.0*Vx[vi];
                    }
                    rhor+=cn*feq[vi]*(avdf[idx_neigh]+m)*weight[vi];
                }
            }

            // 反射边界条件
            for(size_t vi=0;vi<Nv;vi++) {
                double cn=nf.x*Vx[vi]+nf.y*Vy[vi];
                if (cn>0.0) {
                    int idx_neigh = index_vdf(neigh,vi);
                    double m=0.0;
                    if(faces[facei].bc_type==BCType::pressure_far_field){
                        m=2.0*Vx[vi];
                    }
                    double temp = -m-2.0*sqrtPI*rhor;
                    avdf[idx_neigh] = temp;
                }
            }
        }
    }
}
void adjointDVM::getAdjRhs() {
    std::fill(arhs.begin(), arhs.end(), Zero);

    for(size_t cellI=0; cellI<mesh.nOwned; ++cellI) {
        double V = mesh.cells[cellI].V;

        scalar arho = amacro[cellI*Namacro+0];
        scalar aux  = amacro[cellI*Namacro+1];
        scalar auy  = amacro[cellI*Namacro+2];
        scalar atau = amacro[cellI*Namacro+3];
        scalar aqx  = amacro[cellI*Namacro+4];
        scalar aqy  = amacro[cellI*Namacro+5];

        for(size_t vi=0; vi<Nv; ++vi) {
            int idx = index_vdf(cellI, vi);

            scalar udotv = aux*Vx[vi] + auy*Vy[vi];
            scalar qdotv = aqx*Vx[vi] + aqy*Vy[vi];

            scalar coll = arho + udotv + (2.0/3.0*c2[vi]-1.0)*atau + (c2[vi]-2.5)*qdotv;
            arhs[idx]   = cfg.delta * V * coll;
        }
    }
}
void adjointDVM::cellIterAdj(int cellI) {
    if(cellI >= mesh.nOwned) return;

    const auto& cell  = mesh.cells[cellI];
    const auto& faces = mesh.faces;
    double V = cell.V;

    for(size_t vi=0; vi<Nv; ++vi) {
        scalar diag = (cfg.delta + invdt[cellI]) * V;
        scalar res = Zero;

        int idx_cell = index_vdf(cellI, vi);

        for(auto faceI : cell.cell2face) {
            const vector& Sf = faces[faceI].Sf;
            // 注意这里是负号(伴随的迎风方向与正向求解相反)
            double phi = -(Sf.x*Vx[vi] + Sf.y*Vy[vi]);

            int owner = faces[faceI].owner;
            int neigh = faces[faceI].neigh;

            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);

            if(cellI == owner) {
                if(phi >= 0.0) {
                    diag += phi;
                } else {
                    res -= phi * avdf[idx_neigh];
                }
            } else {
                if(phi >= 0.0) {
                    res += phi * avdf[idx_owner];
                } else {
                    diag -= phi;
                }
            }
        }

        res += invdt[cellI] * V * avdf[idx_cell];

        avdf[idx_cell]   = (arhs[idx_cell]   + res) / diag;
    }
}
void adjointDVM::lusgsIterAdj() {
    for(size_t iter=0;iter<5;iter++){
        for(int cellI=0;cellI<mesh.nOwned;cellI++) {
            cellIterAdj(cellI);
        }
        for(int cellI=mesh.nOwned-1;cellI>=0;cellI--) {
            cellIterAdj(cellI);
        }
    }
}
void adjointDVM::step(int iter){
    // rhs 只组装 owned
    getAdjRhs();
    // 本地 LU-SGS + halo 同步
    lusgsIterAdj();
    // 更新边界伪单元
    adjointBoundarySet();
    // 求宏观量并全局归约残差
    updateAdjMacro();
    // 密度守恒性修正
    // massConservation();
    if(rank==0 && (iter%cfg.print_interval==0 || iter ==1)){
        printf("iter %d\t res_aux: %3e\t res_auy: %3e\t res_arho: %3e \t res_atau: %3e\n\n",
            iter, res_aux, res_auy, res_arho, res_atau);
    }
}