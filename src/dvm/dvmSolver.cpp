#include "dvmSolver.h"
#include <algorithm>
#include <cstddef>

dvmSolver::dvmSolver(const Mesh& mesh,
    const SolverConfig& cfg,
    MPI_Comm comm
)
    :mesh(mesh),
    cfg(cfg),
    comm(comm),
    Nv(cfg.Nv)
{
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    initial();
}
void dvmSolver::initial() {
    nonUniform(vx,weight_x,cfg.Nvx,cfg.Lvx);
    nonUniform(vy,weight_y,cfg.Nvy,cfg.Lvy);
    // nonUniform(vz,weight_z,cfg.Nvz,cfg.Lvz);
    uniform(vz,weight_z,cfg.Nvz,cfg.Lvz);
    allocateVelSpace();

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
    exchangeVDF();
}
bool dvmSolver::allocateVelSpace() 
{
    // 分配空间
    if(Nv!=vx.size()*vy.size()*vz.size()){
        printf("Error: Inconsistent velocity space dimensions\n\n");
        return false;
    }
    Vx.resize(Nv);Vy.resize(Nv);Vz.resize(Nv);
    weight.resize(Nv);
    c2.resize(Nv);
    feq.resize(Nv);
    exp_c2.resize(Nv);

    size_t ip=0;
    for(size_t ipz=0; ipz<vz.size();ipz++){
        for (size_t ipy=0; ipy<vy.size();ipy++) {
            for (size_t ipx=0;ipx<vx.size();ipx++) {
                Vx[ip]=vx[ipx];
                Vy[ip]=vy[ipy];
                Vz[ip]=vz[ipz];
                weight[ip]=weight_x[ipx]*weight_y[ipy]*weight_z[ipz];
                ip++;
            }
        }
    }

    // c2, feq ,exp(-c2)
    for (size_t vi=0;vi<Nv;vi++) {
        // c2= vx^2+vy^2+vz^2
        c2[vi]=Vx[vi]*Vx[vi]+Vy[vi]*Vy[vi]+Vz[vi]*Vz[vi];
        // feq= exp(-c2)/pi
        feq[vi]= std::exp(-c2[vi])/PI/sqrtPI;
        // exp(-c2)
        exp_c2[vi]=std::exp(-c2[vi]);
    }
    return true;
}
void uniform(std::vector<double> &vi,
    std::vector<double> &weight_i,
    size_t Nvi,
    double Lvi)
{
    // 分配空间
    vi.resize(Nvi);
    weight_i.resize(Nvi);

    double dx=2.0*Lvi/static_cast<double>(Nvi-1);
    for (size_t ipx=0;ipx<Nvi;ipx++) {
        vi[ipx]=-Lvi+dx*ipx;
        weight_i[ipx]=dx;
    }
}
void nonUniform(std::vector<double> &vi,
                        std::vector<double> &weight_i,
                        size_t Nvi,
                        double Lvi)
{
    // 分配空间
    vi.resize(Nvi);
    weight_i.resize(Nvi);

    for (size_t ipx=0;ipx<Nvi;ipx++) {
        int ix=2*ipx-(Nvi-1);
        vi[ipx]=Lvi*std::pow(ix,3)/std::pow(Nvi-1,3);
        weight_i[ipx]=6.0*Lvi*std::pow(ix,2)/std::pow(Nvi-1,3);
    }
}
void dvmSolver::boundarySet() {
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
            // vector xof=mesh.Cf()[facei]-mesh.C()[owner];
            for(int vi=0;vi<Nv;vi++) {
                double cn=nf.x*Vx[vi]+nf.y*Vy[vi];
                if (cn>0.0) {
                    int idx_owner = index_vdf(owner,vi);
                    int idx_neigh = index_vdf(neigh,vi);
                    vdf[idx_neigh] = vdf[idx_owner];
                    rhor+=cn*exp_c2[vi]*vdf[idx_neigh]*weight[vi]*2.0/PI;
                }
            }
            // 反射边界条件
            double uwx=0.0,uwy=0.0;
            if(faces[facei].bc_type==BCType::pressure_far_field){
                switch (cfg.uwall) {
                    case 0:
                        uwx = nf.y; uwy = -nf.x;
                        break;
                    case 1:
                        // normal x direction
                        uwx = 1.0;
                        break;
                    case 2:
                        // normal y direction
                        uwy = 1.0;
                        break;
                    default:
                        break;
                }
            }
            
            double uwn = uwx*nf.x+uwy*nf.y;

            // 反射边界条件
            for(size_t vi=0;vi<Nv;vi++) {
                double cn=nf.x*Vx[vi]+nf.y*Vy[vi];
                if (cn<0.0) {
                    int idx_neigh = index_vdf(neigh,vi);
                    double udotv = uwx*Vx[vi]+uwy*Vy[vi];
                    scalar temp = rhor -sqrtPI*uwn + 2.0*udotv;
                    vdf[idx_neigh] = temp;
                }
            }
        }
    }
}
void dvmSolver::updateMacro() {
    double up_local[4]={0.0},down_local[4]={0.0};
    for(size_t cellI=0;cellI<mesh.nCells+mesh.nBoundaryFaces;cellI++) {
        // 临时变量
        scalar macro_temp[Nmacro]={Zero};

        // 速度空间积分
        for(size_t vi=0;vi<Nv;vi++){
            int idx = index_vdf(cellI,vi);
            scalar h1 = vdf[idx];
            macro_temp[0]+=h1*feq[vi]*weight[vi];

            macro_temp[1]+=h1*feq[vi]*weight[vi]*Vx[vi];
            macro_temp[2]+=h1*feq[vi]*weight[vi]*Vy[vi];

            macro_temp[3]+=feq[vi]*weight[vi]
                *((2.0*c2[vi]/3.0-1.0)*h1);
            
            macro_temp[4]+=feq[vi]*weight[vi]
                *(c2[vi]-2.5)*h1*Vx[vi];
            macro_temp[5]+=feq[vi]*weight[vi]
                *(c2[vi]-2.5)*h1*Vy[vi];

            macro_temp[6]+=feq[vi]*weight[vi]
                *2.0*(Vx[vi]*Vx[vi]-c2[vi]/3.0)*h1;
            macro_temp[9]+=feq[vi]*weight[vi]
                *2.0*(Vy[vi]*Vy[vi]-c2[vi]/3.0)*h1;
            macro_temp[7]+=feq[vi]*weight[vi]
                *2.0*Vx[vi]*Vy[vi]*h1;
            macro_temp[8]+=feq[vi]*weight[vi]
                *2.0*Vx[vi]*Vy[vi]*h1;
        }
        if(cellI<mesh.nCells){
            double V=mesh.cells[cellI].V;
            for(int i=0;i<4;i++){
                scalar macro_old = macro[cellI*Nmacro+i];
                up_local[i]+=(macro_old-macro_temp[i])*(macro_old-macro_temp[i])*V;
                // down[i]+=macro_temp[i]*macro_temp[i]*V;
                down_local[i]+=macro_old*macro_old*V;
            }
        }

        for(int i=0;i<Nmacro;i++){
            macro[cellI*Nmacro+i]=macro_temp[i];
        }
    }
    double up_global[4]={0.0}, down_global[4]={0.0};
    MPI_Allreduce(up_local,   up_global,   4, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(down_local, down_global, 4, MPI_DOUBLE, MPI_SUM, comm);
    const double eps = 1e-300;
    res_rho = std::sqrt(up_global[0] / std::max(down_global[0], eps));
    res_ux  = std::sqrt(up_global[1] / std::max(down_global[1], eps));
    res_uy  = std::sqrt(up_global[2] / std::max(down_global[2], eps));
    res_tau = std::sqrt(up_global[3] / std::max(down_global[3], eps));
}

void dvmSolver::getRhs() {
    // 清零
    std::fill(rhs.begin(),rhs.end(),Zero);

    const auto& cells = mesh.cells;
    for(size_t cellI=0;cellI<mesh.nOwned;cellI++) {
        double V=cells[cellI].V;
        scalar rho=macro[cellI*Nmacro+0];
        scalar ux=macro[cellI*Nmacro+1];
        scalar uy=macro[cellI*Nmacro+2];
        scalar tau=macro[cellI*Nmacro+3];
        scalar qx=macro[cellI*Nmacro+4];
        scalar qy=macro[cellI*Nmacro+5];

        for(size_t vi=0;vi<Nv;vi++) {
            int idx = index_vdf(cellI,vi);
            // collision
            scalar udotv = ux*Vx[vi]+uy*Vy[vi];
            scalar qdotv = qx*Vx[vi]+qy*Vy[vi];
            scalar coll = rho + 2.0*udotv + (c2[vi]-1.5)*tau + (c2[vi]-2.5)*4.0/15.0*qdotv;
            rhs[idx]   = cfg.delta * V * coll;
        }
    }
}

void dvmSolver::cellIter(size_t cellI) {
    const auto& cell=mesh.cells[cellI];
    const auto& faces=mesh.faces;
    double V=mesh.cells[cellI].V;

    if(cellI>=mesh.nOwned) return;

    for (size_t vi = 0; vi < Nv; ++vi) {
        // 对每个速度点，diag/rhs 必须独立初始化
        scalar diag = (cfg.delta + invdt[cellI]) * V;
        scalar res = Zero;
        int idx_cell = index_vdf(cellI,vi);

        // 对流项：对所有面求和
        for (auto faceI : cell.cell2face) {
            const vector& Sf = faces[faceI].Sf;
            double phi = Sf.x*Vx[vi] + Sf.y*Vy[vi];

            int owner = faces[faceI].owner;
            int neigh = faces[faceI].neigh;

            int idx_owner = index_vdf(owner,vi);
            int idx_negih  = index_vdf(neigh,vi);

            if (cellI == owner) {
                if (phi >= 0.0) {
                    diag += phi;
                } else {
                    res -= phi * vdf[idx_negih];
                }
            } else { // cellI == neighbour
                if (phi >= 0.0) {
                    res += phi * vdf[idx_owner];
                } else {
                    diag -= phi;
                }
            }
        }

        res += invdt[cellI] * V * vdf[idx_cell];

        vdf[idx_cell]   = (rhs[idx_cell]   +  res) / diag;
    }
}
void dvmSolver::exchangeVDF()
{
    haloExchangeCellData(mesh, vdf.data(), Nv, comm);
}
void dvmSolver::lusgsIter() {
    exchangeVDF();
    for(size_t iter=0;iter<5;iter++){
        for(int cellI=0;cellI<mesh.nOwned;cellI++) {
            cellIter(cellI);
        }
        exchangeVDF();
        for(int cellI=mesh.nOwned-1;cellI>=0;cellI--) {
            cellIter(cellI);
        }
        exchangeVDF();
    }
}

// void dvmSolver::massConservation() {
//     // 需要对密度进行守恒性修正
//     double V_sum = 0.0;
//     scalar rho_avr = Zero;
//     for (int cellI = 0; cellI < mesh.nCells; ++cellI) {
//         double V = mesh.cells[cellI].V;
//         scalar rho = macro[cellI*Nmacro+0];
//         rho_avr += rho * V;
//         V_sum   += V;
//     }
//     rho_avr/=V_sum;
//     for(int cellI=0;cellI<mesh.nCells;cellI++){
//         macro[cellI*Nmacro+0] -= rho_avr;
//     }
// }
void dvmSolver::step(int iter) {

    // rhs 只组装 owned
    getRhs();
    // 本地 LU-SGS + halo 同步
    lusgsIter();
    // 更新边界伪单元
    boundarySet();
    // 求宏观量并全局归约残差
    updateMacro();
    // 密度守恒性修正
    // massConservation();
    if(rank==0 && (iter%cfg.print_interval==0 || iter ==1)){
        printf("iter %d\t res_ux: %3e\t res_uy: %3e\t res_rho: %3e \t res_tau: %3e\n\n",
            iter, res_ux, res_uy, res_rho, res_tau);
    }
}