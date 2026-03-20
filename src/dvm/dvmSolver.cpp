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
    Vz(vel.Vz),
    weight(vel.weight),
    c2(vel.c2),
    feq(vel.feq),
    exp_c2(vel.exp_c2),
    weight_macro(vel.weight_macro)
{
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    // 为分布函数分配空间
    // 不仅存储单元内的值，还存储边界面的值
    vdf.resize(Nv*(mesh.nCells+mesh.nBoundaryFaces));
    rhs.resize(Nv*(mesh.nCells+mesh.nBoundaryFaces));
    // macro
    macro.resize((mesh.nCells+mesh.nBoundaryFaces)*Nmacro);
    hot.resize((mesh.nCells+mesh.nBoundaryFaces)*Nhot);
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
        double cn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (cn > 0.0) {
            int idx_owner = index_vdf(owner, vi);
            int idx_neigh = index_vdf(neigh, vi);
            vdf[idx_neigh] = vdf[idx_owner];
            rhor += cn * exp_c2[vi] * vdf[idx_neigh] * weight[vi] * 2.0 / PI;
        }
    }

    double uwn = uwx * nf.x + uwy * nf.y;
    for (int vi = 0; vi < Nv; vi++) {
        double cn = nf.x * Vx[vi] + nf.y * Vy[vi];
        if (cn < 0.0) {
            int idx_neigh = index_vdf(neigh, vi);
            double udotv = uwx * Vx[vi] + uwy * Vy[vi];
            scalar temp = rhor - sqrtPI * uwn + 2.0 * udotv;
            vdf[idx_neigh] = temp;
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
        int idx_vdf = cellI*Nv;
        for(size_t vi=0;vi<Nv;vi++){
            scalar h1 = vdf[idx_vdf+vi];
            for(int i=0;i<Nmacro;i++){
                macro_temp[i]+=h1*weight_macro[vi][i];
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
        scalar rho=macro[cellI*Nmacro+0];
        scalar ux=macro[cellI*Nmacro+1];
        scalar uy=macro[cellI*Nmacro+2];
        scalar tau=macro[cellI*Nmacro+3];
        scalar qx=macro[cellI*Nmacro+4];
        scalar qy=macro[cellI*Nmacro+5];

        int cell_vdf = cellI*Nv;
        for(size_t vi=0;vi<Nv;vi++) {
            // collision
            scalar udotv = ux*Vx[vi]+uy*Vy[vi];
            scalar qdotv = qx*Vx[vi]+qy*Vy[vi];
            scalar coll = rho + 2.0*udotv + (c2[vi]-1.5)*tau + (c2[vi]-2.5)*4.0/15.0*qdotv;
            rhs[cell_vdf+vi]   = cfg.delta * V * coll;
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

        scalar dhdx,dhdy;

        for(size_t vi=0;vi<Nv;vi++) {
            double phi = Sf.x*Vx[vi] + Sf.y*Vy[vi];
            if (phi>0.0) {
                // 计算梯度
                grad(owner,vi,dhdx,dhdy);
                scalar temp = phi*(dhdx*xof.x + dhdy*xof.y);
                rhs[owner*Nv+vi]-= temp;
                rhs[neigh*Nv+vi]+= temp;
            }else{
                grad(neigh,vi,dhdx,dhdy);
                scalar temp = phi*(dhdx*xnf.x + dhdy*xnf.y);
                rhs[owner*Nv+vi]-= temp;
                rhs[neigh*Nv+vi]+= temp;
            }
        }
    }
}
void dvmSolver::cellIter(int cellI) {
    if (cellI >= mesh.nOwned) return;

    const auto& cell  = mesh.cells[cellI];
    const auto& faces = mesh.faces;

    const double V          = cell.V;
    const double alpha_base = (cfg.delta + invdt[cellI]) * V;
    const double beta_dt    = invdt[cellI] * V;

    const auto& face_ids = cell.cell2face;
    const int nf = static_cast<int>(face_ids.size());

    int owner_nb_base[4], neigh_nb_base[4];
    double owner_sfx[4], owner_sfy[4];
    double neigh_sfx[4], neigh_sfy[4];
    int n_owner = 0, n_neigh = 0;

    for (int k = 0; k < nf; ++k) {
        const int faceI = face_ids[k];
        const Face& f = faces[faceI];

        if (cellI == f.owner) {
            owner_sfx[n_owner]    = f.Sf.x;
            owner_sfy[n_owner]    = f.Sf.y;
            owner_nb_base[n_owner] = f.neigh * Nv;
            ++n_owner;
        } else {
            neigh_sfx[n_neigh]    = f.Sf.x;
            neigh_sfy[n_neigh]    = f.Sf.y;
            neigh_nb_base[n_neigh] = f.owner * Nv;
            ++n_neigh;
        }
    }

    const int base_cell = cellI * Nv;
    const double* __restrict__ vxp  = Vx.data();
    const double* __restrict__ vyp  = Vy.data();
    const scalar* __restrict__ rhsp = rhs.data() + base_cell;
    scalar* __restrict__ vcell      = vdf.data() + base_cell;


    for (int vi = 0; vi < Nv; ++vi) {
        const double vx = vxp[vi];
        const double vy = vyp[vi];

        double diag = alpha_base;
        double res  = beta_dt * vcell[vi];

        for (int k = 0; k < n_owner; ++k) {
            const double phi  = owner_sfx[k] * vx + owner_sfy[k] * vy;
            const double ppos = std::max(phi, 0.0);
            const double pneg = std::min(phi, 0.0);
            const scalar vnb  = vdf[owner_nb_base[k] + vi];

            diag += ppos;
            res  -= pneg * vnb;
        }

        for (int k = 0; k < n_neigh; ++k) {
            const double phi  = neigh_sfx[k] * vx + neigh_sfy[k] * vy;
            const double ppos = std::max(phi, 0.0);
            const double pneg = std::min(phi, 0.0);
            const scalar vnb  = vdf[neigh_nb_base[k] + vi];

            res  += ppos * vnb;
            diag -= pneg;
        }

        vcell[vi] = (rhsp[vi] + res) / diag;
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

//     // 注意：如果非结构网格面数可能 > 8，这里必须改成更安全的容器或加断言
//     int    nb_base[8];
//     double sfx[8], sfy[8];
//     int    owner_side[8];   // 1: cellI 是 owner, 0: cellI 是 neigh

//     for (int k = 0; k < nf; ++k) {
//         const int faceI = face_ids[k];
//         const Face& f = faces[faceI];

//         sfx[k] = f.Sf.x;
//         sfy[k] = f.Sf.y;

//         if (cellI == f.owner) {
//             owner_side[k] = 1;
//             nb_base[k]    = f.neigh * Nv;
//         } else {
//             owner_side[k] = 0;
//             nb_base[k]    = f.owner * Nv;
//         }
//     }

//     const int base_cell = cellI * Nv;
//     const double* __restrict__ vxp  = Vx.data();
//     const double* __restrict__ vyp  = Vy.data();
//     const scalar* __restrict__ rhsp = rhs.data() + base_cell;
//     scalar* __restrict__ vcell      = vdf.data() + base_cell;

//     // 告诉编译器：vxp/vyp/rhsp/vcell 都是线性连续的
//     // 邻居数据是 gather，通常是向量化的主要障碍，但至少本 cell 相关部分会更容易优化

//     for (int vi = 0; vi < Nv; ++vi) {
//         const double vx = vxp[vi];
//         const double vy = vyp[vi];

//         double diag = alpha_base;
//         double res  = beta_dt * vcell[vi];

//         // 对每个面做固定形式更新，减少 if-else 分支
//         for (int k = 0; k < nf; ++k) {
//             const double phi  = sfx[k] * vx + sfy[k] * vy;
//             const double ppos = std::max(phi, 0.0);
//             const double pneg = std::min(phi, 0.0);

//             const scalar vnb = vdf[nb_base[k] + vi];

//             if (owner_side[k]) {
//                 diag += ppos;
//                 res  -= pneg * vnb;
//             } else {
//                 res  += ppos * vnb;
//                 diag -= pneg;
//             }
//         }

//         vcell[vi] = (rhsp[vi] + res) / diag;
//     }
// }

// void dvmSolver::cellIter(int cellI) {
//     if(cellI >= mesh.nOwned) return;
//     const auto& cell  = mesh.cells[cellI];
//     const auto& faces = mesh.faces;

//     const double V = cell.V;
//     const double alpha_base = (cfg.delta + invdt[cellI]) * V;
//     const double beta_dt    = invdt[cellI] * V;

//     const auto& face_ids = cell.cell2face;
//     const int nf = face_ids.size();

//     // 预提取当前cell每个face的常量信息
//     int neigh_id[8];
//     double sfx[8], sfy[8];
//     int orient[8]; // +1: cellI == owner, -1: cellI == neigh

//     for (int k = 0; k < nf; ++k) {
//         int faceI = face_ids[k];
//         const Face& f = faces[faceI];

//         sfx[k] = f.Sf.x;
//         sfy[k] = f.Sf.y;

//         if (cellI == f.owner) {
//             orient[k]  = +1;
//             neigh_id[k] = f.neigh;
//         } else {
//             orient[k]  = -1;
//             neigh_id[k] = f.owner;
//         }
//     }

//     const int base_cell = cellI * Nv;

//     for (int vi = 0; vi < Nv; ++vi) {
//         scalar diag = alpha_base;
//         scalar res  = beta_dt * vdf[base_cell + vi];

//         const double vx = Vx[vi];
//         const double vy = Vy[vi];

//         for (int k = 0; k < nf; ++k) {
//             const double phi = sfx[k] * vx + sfy[k] * vy;
//             const int idx_nb = neigh_id[k] * Nv + vi;

//             if (orient[k] > 0) {
//                 if (phi >= 0.0) diag += phi;
//                 else            res  -= phi * vdf[idx_nb];
//             } else {
//                 if (phi >= 0.0) res  += phi * vdf[idx_nb];
//                 else            diag -= phi;
//             }
//         }
//         vdf[base_cell + vi] = (rhs[base_cell + vi] + res) / diag;
//     }
//  }

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

void dvmSolver::grad(int cellI,int vi, double& gradx, double& grady){
    const auto& cell  = mesh.cells[cellI];
    const auto& faces = mesh.faces;

    const double V = cell.V;
    if (V <= 1e-30) {
        gradx = 0.0;
        grady = 0.0;
        return;
    }
    const auto& face_ids = cell.cell2face;
    const int nf = static_cast<int>(face_ids.size());

    double dh = 0.0;
    double b[2] = {0.0,0.0};
    for (int k = 0; k < nf; ++k) {
        const int faceI = face_ids[k];
        const Face& f = faces[faceI];

        int owner = f.owner;
        int neigh = f.neigh;

        if(cellI==owner){
            dh = vdf[neigh*Nv + vi] - vdf[owner*Nv + vi];
        }else{
            dh = vdf[owner*Nv + vi] - vdf[neigh*Nv + vi];
        }
        b[0] += dh * cell.dxyz[k][0];
        b[1] += dh * cell.dxyz[k][1];
    }
    const auto& invA = cell.invA;
    gradx = invA[0]*b[0] + invA[1]*b[1];
    grady = invA[1]*b[0] + invA[2]*b[1];
}