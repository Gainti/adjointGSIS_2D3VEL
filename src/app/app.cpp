#include "app.h"


#include "timer.hpp"
#include "utils.h"
#include "output.h"

bool initializeApp(int argc, char** argv, AppContext& ctx) {
    MPI_Comm_rank(ctx.comm, &ctx.rank);
    MPI_Comm_size(ctx.comm, &ctx.size);

    if(ctx.rank == 0) {
        printf("\nInitializing application...\n");
    }

    if (!readCaseConfig(argc, argv, ctx.case_info, ctx.cfg, ctx.rank)) {
        return false;
    }

    if(ctx.rank == 0) {
        printf("\nReading mesh...\n");
    }
    if (!readMesh(ctx.case_info.mesh_file,
                  ctx.globalMesh,
                  ctx.localMesh,
                  ctx.comm,
                  ctx.rank)) {
        return false;
    }

    if(ctx.rank == 0) {
        printf("\nBuilding velocity space...\n");
    }
    if (!buildVelocitySpace(ctx)) {
        return false;
    }

    return true;
}

bool runValidationTasks(AppContext& ctx) {

    // message("mesh deform",rank);
    // std::vector<int> boundaryPts;
    // collectAllBoundaryPoints(globalMesh, boundaryPts);
    
    // std::vector<BoundaryNodeDisplacement> bdisp;
    // bdisp.reserve(boundaryPts.size());
    
    // double scale = 0.8;
    // for (int pid : boundaryPts)
    // {
    //     const auto& point = globalMesh.points[pid];
    //     vector d(0.0, (scale-1.0)*point.y, 0.0);
    //     bdisp.push_back({pid, d});
    // }
    
    // deformMeshSpring(globalMesh,localMesh, bdisp, MPI_COMM_WORLD, 500, 1e-10, 1.8);
    // MPI_Barrier(MPI_COMM_WORLD);

    // 验证最小二乘法计算梯度的正确性
    // validateLeastSquaresLinearField(localMesh, MPI_COMM_WORLD, 1, -2, 0.5);

    // 几何导数链验证
    // runGeometryChainValidation(localMesh,1.0e-7);

    // 检查边界节点移动后有限差分导数和伴随导数的一致性
    // double eps_scale = 1e-4;
    // validateOneBoundaryPoint(localMesh,vel,scfg, 29, 1, eps_scale, MPI_COMM_WORLD);
    // validateOneBoundaryPoint(localMesh,vel,scfg, 3, 1, eps_scale, MPI_COMM_WORLD);  

    // 检查边界缩放后有限差分导数和伴随导数的一致性
    // validateYscaleboundary(globalMesh,localMesh,vel,scfg, MPI_COMM_WORLD);

    // 检查剪切壁面按形函数沿y方向变化的有限差分导数和伴随导数的一致性
    // validatePressureFarFieldBoundaryYMode(localMesh,vel,scfg,MPI_COMM_WORLD);
    return true;
}
bool runForwardSolve(AppContext& ctx) {
    double t0, t1;
    if(ctx.rank == 0) {
        printf("\nRunning forward solve...\n");
    }
    const Mesh& globalMesh = ctx.globalMesh;
    const Mesh& localMesh = ctx.localMesh;
    const VelocitySpace& vel = ctx.vel;

    if(ctx.dvm == nullptr){
        ctx.dvm = new dvmSolver(ctx.localMesh, ctx.vel, ctx.cfg, ctx.comm);
    }
    dvmSolver& dvm = *ctx.dvm;
    t0 = MPI_Wtime();
    for(int iter=1; iter<ctx.cfg.max_iter; iter++){
        dvm.step(iter);
        if(dvm.res_ux < ctx.cfg.tol 
            && dvm.res_uy < ctx.cfg.tol 
            && dvm.res_rho < ctx.cfg.tol) {
            if (ctx.rank == 0) {
                printf("DVM converged at iteration %d\n\n", iter);
            }
            break;
        }
    }
    t1 = MPI_Wtime();
    report_stage_time("dvm", t1 - t0, MPI_COMM_WORLD);
    // dvm.reportProfile();
    return true;
}
bool runAdjointSolve(AppContext& ctx) {
    if(ctx.rank == 0) {
        printf("\nRunning adjoint solve...\n");
    }
    const Mesh& globalMesh = ctx.globalMesh;
    const Mesh& localMesh = ctx.localMesh;
    const VelocitySpace& vel = ctx.vel;

    if(ctx.adj == nullptr){
        ctx.adj = new adjointDVM(ctx.localMesh, ctx.vel, ctx.cfg, ctx.comm);
    }
    adjointDVM& adj = *ctx.adj;

    double t0 = MPI_Wtime();
    for(int iter=1;iter<ctx.cfg.max_iter;iter++){
        adj.step(iter);
        if(adj.res_aux < ctx.cfg.tol 
            && adj.res_auy < ctx.cfg.tol 
            && adj.res_arho < ctx.cfg.tol) {
            if (ctx.rank == 0) {
                printf("Adjoint DVM converged at iteration %d\n\n", iter);
            }
            break;
        }
    }
    double t1 = MPI_Wtime();
    report_stage_time("adjoint", t1 - t0, MPI_COMM_WORLD);
    return true;
}
bool writeOutputs(AppContext& ctx) {
    if(ctx.rank == 0) {
        printf("\nWriting outputs...\n");
    }
    const std::string output_dir = ctx.case_info.output_dir;
    const Mesh& globalMesh = ctx.globalMesh;
    const Mesh& localMesh = ctx.localMesh;
    if(ctx.dvm != nullptr){
        std::string filename = utils::join_path(output_dir, "macro");
        dvmSolver& dvm = *ctx.dvm;
        write_tecplot(filename,globalMesh,localMesh,dvm.macro,Nmacro,MPI_COMM_WORLD);
    }
    if(ctx.adj != nullptr){
        std::string filename = utils::join_path(output_dir, "adj_macro");
        adjointDVM& adj = *ctx.adj;
        write_tecplot(filename,globalMesh,localMesh,adj.amacro,Namacro,MPI_COMM_WORLD);
    }
    return true;
}

bool runApp(AppContext& ctx) {
    runForwardSolve(ctx);
    runAdjointSolve(ctx);
    writeOutputs(ctx);
    return true;
}