#include "adjointDVM.h"

#include <algorithm>
#include <cmath>

adjointDVM::adjointDVM(const Mesh& mesh,
    const VelocitySpace& vel,
    const SolverConfig& cfg,
    MPI_Comm comm
)
    :mesh(mesh),
    cfg(cfg),
    comm(comm),
    Nv(cfg.Nv),
    Vx(vel.Vx),
    Vy(vel.Vy),
    v2(vel.v2),
    feq(vel.feq),
    weight(vel.weight),
    weight_macro(vel.weight_macro),
    weight_coll(vel.weight_coll)
{
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    avdf.resize(Nvdf*Nv * (mesh.nCells + mesh.nBoundaryFaces), Zero);
    arhs.resize(Nvdf*Nv * (mesh.nCells + mesh.nBoundaryFaces), Zero);
    amacro.resize((mesh.nCells + mesh.nBoundaryFaces) * Namacro, Zero);

    // pseudo time step
    invdt.resize(mesh.nCells, 0.0);
    for(int celli=0;celli<mesh.nCells;celli++){
        invdt[celli]=0.0;
    }

    // fill avdf with zero
    std::fill(avdf.begin(),avdf.end(),Zero);

    adjointBoundarySet();
    updateAdjMacro();
}

void adjointDVM::updateAdjMacro() {
    double up_local[4]={0.0},down_local[4]={0.0};
    for(size_t cellI=0; cellI<mesh.nCells+mesh.nBoundaryFaces; ++cellI) {
        scalar amacro_t[6] = {Zero};

        for(size_t vi=0; vi<Nv; ++vi) {
            int idx = index_vdf(cellI, vi);
            scalar phi1 = avdf[idx];
            scalar phi2 = avdf[idx+1];
            double fw = feq[vi] * weight[vi];
            for(int i=0;i<Nmacro;i++){
                amacro_t[i]+=fw*(phi1*weight_coll[vi][i*2]
                    +phi2*weight_coll[vi][i*2+1]);
            }
        }
        if(cellI<mesh.nCells){
            double V=mesh.cells[cellI].V;
            for(int i=0;i<4;i++){
                scalar amacro_old = amacro[cellI*Namacro+i];
                up_local[i]+=(amacro_old-amacro_t[i])*(amacro_old-amacro_t[i])*V;
                down_local[i]+=amacro_t[i]*amacro_t[i]*V;
            }
        }
        for(int k=0;k<Namacro;k++) {
            amacro[cellI*Namacro+k] = amacro_t[k];
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

void adjointDVM::diffuseWall(int facei) {
    const auto& face = mesh.faces[facei];
    int owner = face.owner;
    int neigh = face.neigh;

    const vector& Sf = face.Sf;
    vector nf = Sf / Sf.mag();

    scalar rhor = Zero;
    for (int vi = 0; vi < Nv; vi++) {
        double vn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (vn < 0.0) {
            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);
            // zero-gradient
            avdf[idx_neigh+0] = avdf[idx_owner+0];
            avdf[idx_neigh+1] = avdf[idx_owner+1];
            double m = 0.0;
            rhor += vn * (avdf[idx_neigh+0] + avdf[idx_neigh+1] + m) 
            * feq[vi]*weight[vi];
        }
    }

    for (int vi = 0; vi < Nv; vi++) {
        double vn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (vn > 0.0) {
            int idx_neigh = index_vdf(neigh, vi);
            double m = 0.0;
            double temp = -m - 2.0 * sqrtPI * rhor;
            avdf[idx_neigh+0] = temp;
            avdf[idx_neigh+1] = 0.0;
        }
    }
}
void adjointDVM::diffuseWallwithObject(int facei) {
    const auto& face = mesh.faces[facei];
    int owner = face.owner;
    int neigh = face.neigh;

    const vector& Sf = face.Sf;
    vector nf = Sf / Sf.mag();

    scalar rhor = Zero;
    for (int vi = 0; vi < Nv; vi++) {
        double vn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (vn < 0.0) {
            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);
            avdf[idx_neigh+0] = avdf[idx_owner+0];
            avdf[idx_neigh+1] = avdf[idx_owner+1];
            double m = objectiveWeight(Vx[vi], Vy[vi]);
            rhor += vn * (avdf[idx_neigh+0] + avdf[idx_neigh+1] + m) 
            * feq[vi] * weight[vi];
        }
    }

    for (int vi = 0; vi < Nv; vi++) {
        double vn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (vn > 0.0) {
            int idx_neigh = index_vdf(neigh, vi);
            double m = objectiveWeight(Vx[vi], Vy[vi]);
            double temp = -m - 2.0 * sqrtPI * rhor;
            avdf[idx_neigh+0] = temp;
            avdf[idx_neigh+1] = 0.0;
        }
    }
}
void adjointDVM::adjointBoundarySet() {
    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; facei++) {
        const auto bcType = mesh.faces[facei].bc_type;
        if (bcType == BCType::wall) {
            diffuseWall(facei);
        } else if (bcType == BCType::pressure_far_field) {
            diffuseWallwithObject(facei);
        }
    }
}
void adjointDVM::getAdjRhs() {
    std::fill(arhs.begin(), arhs.end(), Zero);
    const auto& cells = mesh.cells;
    const auto& faces = mesh.faces;

    for(size_t cellI=0; cellI<mesh.nOwned; ++cellI) {
        double V = mesh.cells[cellI].V;
        for(size_t vi=0; vi<Nv; ++vi) {
            int idx_vdf = index_vdf(cellI, vi);
            // collision
            scalar coll[2]={Zero};
            for(int i=0;i<Nmacro;i++){
                coll[0]+=weight_macro[vi][i*2]*amacro[cellI*Nmacro+i];
                coll[1]+=weight_macro[vi][i*2+1]*amacro[cellI*Nmacro+i];
            }
            arhs[idx_vdf]   = cfg.delta * V * coll[0];
            arhs[idx_vdf+1] = cfg.delta * V * coll[1];
        }
    }

    // 二阶迎风
    for(int faceI=0;faceI<mesh.nInternalFaces;faceI++) {
        int owner=faces[faceI].owner;
        int neigh=faces[faceI].neigh;
        const vector& Sf=faces[faceI].Sf;
        double magSf=Sf.mag();
        vector nf= Sf/magSf;

        vector xof = faces[faceI].Cf -cells[owner].C;
        vector xnf = faces[faceI].Cf -cells[neigh].C;

        scalar dphi[Nvdf*dim];// (i,j) i*dim+j

        for(size_t vi=0;vi<Nv;vi++) {
            double phi = -(Sf.x*Vx[vi] + Sf.y*Vy[vi]);
            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);
            scalar temp[Nvdf];
            if (phi>0.0) {
                // 计算梯度
                grad(owner,vi,dphi,Nvdf);
                temp[0] = phi*(dphi[0]*xof.x + dphi[1]*xof.y);
                temp[1] = phi*(dphi[2]*xof.x + dphi[3]*xof.y);
            }else{
                grad(neigh,vi,dphi,Nvdf);
                temp[0] = phi*(dphi[0]*xnf.x + dphi[1]*xnf.y);
                temp[1] = phi*(dphi[2]*xnf.x + dphi[3]*xnf.y);
            }
            for(int i=0;i<Nvdf;i++){
                arhs[idx_owner+i]-= temp[i];
                arhs[idx_neigh+i]+= temp[i];
            }
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
        scalar res[2] = {Zero};

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
                    res[0] -= phi * avdf[idx_neigh+0];
                    res[1] -= phi * avdf[idx_neigh+1];
                }
            } else {
                if(phi >= 0.0) {
                    res[0] += phi * avdf[idx_owner+0];
                    res[1] += phi * avdf[idx_owner+1];
                } else {
                    diag -= phi;
                }
            }
        }

        res[0] += invdt[cellI] * V * avdf[idx_cell+0];
        res[1] += invdt[cellI] * V * avdf[idx_cell+1];

        avdf[idx_cell+0] = (arhs[idx_cell+0]   + res[0]) / diag;
        avdf[idx_cell+1] = (arhs[idx_cell+1]   + res[1]) / diag;
    }
}
void adjointDVM::lusgsIterAdj() {
    for(size_t iter=0;iter<2;iter++){
        for(int cellI=0;cellI<mesh.nOwned;cellI++) {
            cellIterAdj(cellI);
        }
        for(int cellI=mesh.nOwned-1;cellI>=0;cellI--) {
            cellIterAdj(cellI);
        }
    }
}

void adjointDVM::massConservation() {
    // 需要对密度进行守恒性修正
    double V_sum = 0.0;
    scalar rho_avr = Zero;
    for (int cellI = 0; cellI < mesh.nCells; ++cellI) {
        double V = mesh.cells[cellI].V;
        scalar rho = amacro[cellI*Namacro+0];
        rho_avr += rho * V;
        V_sum   += V;
    }
    rho_avr/=V_sum;
    for(int cellI=0;cellI<mesh.nCells;cellI++){
        amacro[cellI*Namacro+0] -= rho_avr;
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
    massConservation();
    if(rank==0 && (iter%cfg.print_interval==0 || iter ==1)){
        printf("iter %d\t res_aux: %3e\t res_auy: %3e\t res_arho: %3e \t res_atau: %3e\n\n",
            iter, res_aux, res_auy, res_arho, res_atau);
    }
}
void adjointDVM::grad(int cellI,int vi, scalar* gradh,int nvdf){
    const auto& cell  = mesh.cells[cellI];
    const auto& faces = mesh.faces;

    if(nvdf>Nvdf || nvdf<=0){
        printf("Error: nvdf exceeds the maximum value.\n");
        return;
    }

    const double V = cell.V;
    if (V <= 1e-30) {
        for(int i=0;i<nvdf;i++){
            for(int j=0;j<dim;j++){
                gradh[i*dim+j] = Zero;
            }
        }
        return;
    }
    const auto& face_ids = cell.cell2face;
    const int nf = static_cast<int>(face_ids.size());

    scalar dh[nvdf] = {Zero};
    scalar b[nvdf*dim] = {Zero};
    for (int k = 0; k < nf; ++k) {
        const int faceI = face_ids[k];
        const Face& f = faces[faceI];

        int owner = f.owner;
        int neigh = f.neigh;

        int idx_owner = index_vdf(owner, vi);
        int idx_neigh = index_vdf(neigh, vi);
        if(cellI==owner){
            for(int i=0;i<nvdf;i++){
                dh[i] = avdf[idx_neigh+i] - avdf[idx_owner+i];
            }
        }else{
            for(int i=0;i<nvdf;i++){
                dh[i] = avdf[idx_owner+i] - avdf[idx_neigh+i];
            }
        }
        for(int i=0;i<nvdf;i++){
            for(int j=0;j<dim;j++){
                b[i*dim+j] += dh[i] * cell.dxyz[k][j];
            }
        }
    }
    const auto& invA = cell.invA;
    for(int i=0;i<nvdf;i++){
        gradh[i*dim+0] = invA[0]*b[i*dim+0] + invA[1]*b[i*dim+1];
        gradh[i*dim+1] = invA[1]*b[i*dim+0] + invA[2]*b[i*dim+1];
    }
}