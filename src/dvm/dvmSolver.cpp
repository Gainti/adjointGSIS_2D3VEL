#include "dvmSolver.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

dvmSolver::dvmSolver(const Mesh& mesh,
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
    weight(vel.weight),
    v2(vel.v2),
    feq(vel.feq),
    weight_macro(vel.weight_macro),
    weight_coll(vel.weight_coll)
{
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    // 为分布函数分配空间
    // 不仅存储单元内的值，还存储边界面的值
    vdf.resize(Nvdf*Nv*(mesh.nCells+mesh.nBoundaryFaces));
    rhs.resize(Nvdf*Nv*(mesh.nCells+mesh.nBoundaryFaces));
    // macro
    macro.resize((mesh.nCells+mesh.nBoundaryFaces)*Nmacro);
    beta.resize(mesh.nCells);
    invdt.resize(mesh.nCells);

    // set zero
    std::fill(vdf.begin(),vdf.end(),Zero);

    // beta and invdt
    for(int celli=0;celli<mesh.nCells;celli++){
        beta[celli]=0.0;
        invdt[celli]=0.0;
    }

    boundarySet();
    updateMacro();
    initHaloWorkspace(mesh, halo_ws, Nv);

}

void dvmSolver::diffuseWall(int facei, double uwx, double uwy) {
    const auto& face = mesh.faces[facei];
    int owner = face.owner;
    int neigh = face.neigh;

    const vector& Sf = face.Sf;
    vector nf = Sf / Sf.mag();

    scalar rhor = Zero;
    for (int vi = 0; vi < Nv; vi++) {
        double vn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (vn > 0.0) {
            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);
            // 零梯度近似
            vdf[idx_neigh+0] = vdf[idx_owner+0];
            vdf[idx_neigh+1] = vdf[idx_owner+1];

            double h1 = vdf[idx_neigh+0];
            rhor += vn * h1 * feq[vi] * weight[vi];
        }
    }
    rhor*= 2.0*sqrtPI;

    double uwn = uwx * nf.x + uwy * nf.y;
    for (int vi = 0; vi < Nv; vi++) {
        double vn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (vn < 0.0) {
            int idx_neigh = index_vdf(neigh, vi);
            double udotv = uwx * Vx[vi] + uwy * Vy[vi];
            scalar temp = rhor - sqrtPI * uwn + 2.0 * udotv;
            vdf[idx_neigh+0] = temp;
            vdf[idx_neigh+1] = temp;
        }
    }
}

void dvmSolver::boundarySet() {
    ScopedTimer timer(profiler, DvmTimerID::BoundarySet);

    for (int facei = mesh.nInternalFaces; facei < mesh.nFaces; facei++) {
        const auto bcType = mesh.faces[facei].bc_type;
        if (bcType == BCType::wall) {
            diffuseWall(facei, 0.0, 0.0);
        } else if (bcType == BCType::pressure_far_field) {
            double uw[2]={0.0, 0.0};
            switch (cfg.uwall) {
                case 0:{
                    const vector& Sf = mesh.faces[facei].Sf;
                    vector nf = Sf / Sf.mag();
                    uw[0] = nf.y;
                    uw[1] = -nf.x;
                    break;
                }
                case 1:{
                    uw[0] = 1.0;
                    break;
                }
                case 2:{
                    uw[1] = 1.0;
                    break;
                }
                default:
                    break;
            }
            diffuseWall(facei, uw[0], uw[1]);
        }
    }
}
void dvmSolver::updateMacro() {
    ScopedTimer timer_total(profiler, DvmTimerID::UpdateMacroTotal);

    double up_local[4]={0.0},down_local[4]={0.0};
    for(size_t cellI=0;cellI<mesh.nCells+mesh.nBoundaryFaces;cellI++) {
        // 临时变量
        scalar macro_temp[Nmacro]={Zero};

        // 速度空间积分
        for(size_t vi=0;vi<Nv;vi++){
            int idx_vdf = index_vdf(cellI, vi);
            scalar h1 = vdf[idx_vdf+0];
            scalar h2 = vdf[idx_vdf+1];
            double fw = feq[vi] * weight[vi];
            for(int i=0;i<Nmacro;i++){
                macro_temp[i]+=fw*(h1*weight_macro[vi][i*2]+h2*weight_macro[vi][i*2+1]);
            }
        }
        if(cellI<mesh.nCells){
            double V=mesh.cells[cellI].V;
            for(int i=0;i<4;i++){
                scalar macro_old = macro[cellI*Nmacro+i];
                up_local[i]+=(macro_old-macro_temp[i])*(macro_old-macro_temp[i])*V;
                down_local[i]+=macro_temp[i]*macro_temp[i]*V;
                // down_local[i]+=macro_old*macro_old*V;
            }
        }

        for(int i=0;i<Nmacro;i++){
            macro[cellI*Nmacro+i]=macro_temp[i];
        }
    }
    double up_global[4]={0.0}, down_global[4]={0.0};
    {
        ScopedTimer timer_allreduce(profiler, DvmTimerID::UpdateMacroAllreduce);
        MPI_Allreduce(up_local,   up_global,   4, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(down_local, down_global, 4, MPI_DOUBLE, MPI_SUM, comm);
    }
    const double eps = 1e-300;
    res_rho = std::sqrt(up_global[0] / std::max(down_global[0], eps));
    res_ux  = std::sqrt(up_global[1] / std::max(down_global[1], eps));
    res_uy  = std::sqrt(up_global[2] / std::max(down_global[2], eps));
    res_tau = std::sqrt(up_global[3] / std::max(down_global[3], eps));
}

void dvmSolver::getRhs() {
    ScopedTimer timer(profiler, DvmTimerID::GetRhs);
    const auto& cells = mesh.cells;
    const auto& faces = mesh.faces;

    for(size_t cellI=0;cellI<mesh.nOwned;cellI++) {
        double V=cells[cellI].V;
        for(size_t vi=0;vi<Nv;vi++) {
            int idx_vdf = index_vdf(cellI, vi);
            // collision
            scalar coll[2]={Zero};
            for(int i=0;i<Nmacro;i++){
                coll[0]+=weight_coll[vi][i*2]*macro[cellI*Nmacro+i];
                coll[1]+=weight_coll[vi][i*2+1]*macro[cellI*Nmacro+i];
            }
            rhs[idx_vdf+0]   = cfg.delta * V * coll[0];
            rhs[idx_vdf+1]   = cfg.delta * V * coll[1];
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

        scalar dh[Nvdf*dim];// (i,j) i*dim+j

        for(size_t vi=0;vi<Nv;vi++) {
            double phi = Sf.x*Vx[vi] + Sf.y*Vy[vi];
            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);
            scalar temp[Nvdf];
            if (phi>0.0) {
                // 计算梯度
                grad(owner,vi,dh,Nvdf);
                temp[0] = phi*(dh[0]*xof.x + dh[1]*xof.y);
                temp[1] = phi*(dh[2]*xof.x + dh[3]*xof.y);
            }else{
                grad(neigh,vi,dh,Nvdf);
                temp[0] = phi*(dh[0]*xnf.x + dh[1]*xnf.y);
                temp[1] = phi*(dh[2]*xnf.x + dh[3]*xnf.y);
            }
            for(int i=0;i<Nvdf;i++){
                rhs[idx_owner+i]-= temp[i];
                rhs[idx_neigh+i]+= temp[i];
            }
        }
    }
}

void dvmSolver::cellIter(int cellI) {
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
            double phi = Sf.x*Vx[vi] + Sf.y*Vy[vi];

            int owner = faces[faceI].owner;
            int neigh = faces[faceI].neigh;

            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);

            if(cellI == owner) {
                if(phi >= 0.0) {
                    diag += phi;
                } else {
                    res[0] -= phi * vdf[idx_neigh+0];
                    res[1] -= phi * vdf[idx_neigh+1];
                }
            } else {
                if(phi >= 0.0) {
                    res[0] += phi * vdf[idx_owner+0];
                    res[1] += phi * vdf[idx_owner+1];
                } else {
                    diag -= phi;
                }
            }
        }

        res[0] += invdt[cellI] * V * vdf[idx_cell+0];
        res[1] += invdt[cellI] * V * vdf[idx_cell+1];

        vdf[idx_cell+0] = (rhs[idx_cell+0] + res[0]) / diag;
        vdf[idx_cell+1] = (rhs[idx_cell+1] + res[1]) / diag;
    }
}
// void dvmSolver::cellIter(int cellI) {
//     if (cellI >= mesh.nOwned) return;

//     const auto& cell  = mesh.cells[cellI];
//     const auto& faces = mesh.faces;

//     const double V          = cell.V;
//     const double alpha_base = (cfg.delta + invdt[cellI]) * V;
//     const double beta_dt    = invdt[cellI] * V;

//     const auto& face_ids = cell.cell2face;
//     const int nf = static_cast<int>(face_ids.size());

//     int owner_nb_base[4], neigh_nb_base[4];
//     double owner_sfx[4], owner_sfy[4];
//     double neigh_sfx[4], neigh_sfy[4];
//     int n_owner = 0, n_neigh = 0;

//     for (int k = 0; k < nf; ++k) {
//         const int faceI = face_ids[k];
//         const Face& f = faces[faceI];

//         if (cellI == f.owner) {
//             owner_sfx[n_owner]    = f.Sf.x;
//             owner_sfy[n_owner]    = f.Sf.y;
//             owner_nb_base[n_owner] = f.neigh * Nv;
//             ++n_owner;
//         } else {
//             neigh_sfx[n_neigh]    = f.Sf.x;
//             neigh_sfy[n_neigh]    = f.Sf.y;
//             neigh_nb_base[n_neigh] = f.owner * Nv;
//             ++n_neigh;
//         }
//     }

//     const int base_cell = cellI * Nv;
//     const double* __restrict__ vxp  = Vx.data();
//     const double* __restrict__ vyp  = Vy.data();
//     const scalar* __restrict__ rhsp = rhs.data() + base_cell;
//     scalar* __restrict__ vcell      = vdf.data() + base_cell;


//     for (int vi = 0; vi < Nv; ++vi) {
//         const double vx = vxp[vi];
//         const double vy = vyp[vi];

//         double diag = alpha_base;
//         double res  = beta_dt * vcell[vi];

//         for (int k = 0; k < n_owner; ++k) {
//             const double phi  = owner_sfx[k] * vx + owner_sfy[k] * vy;
//             const double ppos = std::max(phi, 0.0);
//             const double pneg = std::min(phi, 0.0);
//             const scalar vnb  = vdf[owner_nb_base[k] + vi];

//             diag += ppos;
//             res  -= pneg * vnb;
//         }

//         for (int k = 0; k < n_neigh; ++k) {
//             const double phi  = neigh_sfx[k] * vx + neigh_sfy[k] * vy;
//             const double ppos = std::max(phi, 0.0);
//             const double pneg = std::min(phi, 0.0);
//             const scalar vnb  = vdf[neigh_nb_base[k] + vi];

//             res  += ppos * vnb;
//             diag -= pneg;
//         }

//         vcell[vi] = (rhsp[vi] + res) / diag;
//     }
// }

void dvmSolver::sweepCells(const std::vector<int>& cellList, bool forward)
{
    if (forward) {
        for (int idx = 0; idx < (int)cellList.size(); ++idx) {
            cellIter(cellList[idx]);
        }
    } else {
        for (int idx = (int)cellList.size() - 1; idx >= 0; --idx) {
            cellIter(cellList[idx]);
        }
    }
}

void dvmSolver::lusgsIter()
{
    ScopedTimer timer_total(profiler, DvmTimerID::LusgsTotal);

    for (size_t iter = 0; iter < 2; ++iter) {

        // ---------- forward sweep ----------
        {
            ScopedTimer timer_fw(profiler, DvmTimerID::LusgsForwardTotal);

            HaloExchangeStats stats{};

            haloExchangeBegin(mesh, halo_ws, vdf.data(), Nv, comm, &stats);

            {
                ScopedTimer timer_interior(profiler, DvmTimerID::LusgsForwardInterior);
                sweepCells(mesh.interiorCells, true);
            }

            {
                ScopedTimer timer_block(profiler, DvmTimerID::LusgsForwardBlockWait);
                haloExchangeEnd(mesh, halo_ws, vdf.data(), Nv, comm, &stats);
            }

            profiler.addHaloStats(stats);
            sweepCells(mesh.boundaryCells, true);
        }

        // ---------- backward sweep ----------
        {
            ScopedTimer timer_bw(profiler, DvmTimerID::LusgsBackwardTotal);

            HaloExchangeStats stats{};

            haloExchangeBegin(mesh, halo_ws, vdf.data(), Nv, comm, &stats);

            {
                ScopedTimer timer_interior(profiler, DvmTimerID::LusgsBackwardInterior);
                sweepCells(mesh.interiorCells, false);
            }

            {
                ScopedTimer timer_block(profiler, DvmTimerID::LusgsBackwardBlockWait);
                haloExchangeEnd(mesh, halo_ws, vdf.data(), Nv, comm, &stats);
            }

            profiler.addHaloStats(stats);
            sweepCells(mesh.boundaryCells, false);
        }
    }
}

void dvmSolver::massConservation() {
    // 需要对密度进行守恒性修正
    double V_sum = 0.0;
    scalar rho_avr = Zero;
    for (int cellI = 0; cellI < mesh.nCells; ++cellI) {
        double V = mesh.cells[cellI].V;
        scalar rho = macro[cellI*Nmacro+0];
        rho_avr += rho * V;
        V_sum   += V;
    }
    rho_avr/=V_sum;
    for(int cellI=0;cellI<mesh.nCells;cellI++){
        macro[cellI*Nmacro+0] -= rho_avr;
    }
}
void dvmSolver::step(int iter) {
    ScopedTimer timer(profiler, DvmTimerID::StepTotal);
    
    // rhs 只组装 owned
    getRhs();
    // 本地 LU-SGS + halo 同步
    lusgsIter();
    // 更新边界伪单元
    boundarySet();
    // 求宏观量并全局归约残差
    updateMacro();
    // 密度守恒性修正
    massConservation();
    if(rank==0 && (iter%cfg.print_interval==0 || iter ==1)){
        printf("iter %d\t res_ux: %3e\t res_uy: %3e\t res_rho: %3e \t res_tau: %3e\n\n",
            iter, res_ux, res_uy, res_rho, res_tau);
    }
}

void dvmSolver::grad(int cellI,int vi, scalar* gradh,int nvdf){
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
                dh[i] = vdf[idx_neigh+i] - vdf[idx_owner+i];
            }
        }else{
            for(int i=0;i<nvdf;i++){
                dh[i] = vdf[idx_owner+i] - vdf[idx_neigh+i];
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