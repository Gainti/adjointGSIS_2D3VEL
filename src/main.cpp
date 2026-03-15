#include "config.h"
#include "utils.h"

#include <cstdio>
#include <string>

#include <mpi.h>
#include "defs.h"
#include "mesh.h"
#include "dvmSolver.h"
#include "timer.hpp"
#include "output.h"

#include "partition.h"


#include "checkMesh.hpp"

#include "meshDeform.h"
#include "unordered_set"

#include "boundary_sensitivity.h"

#include "object.hpp"

static void usage() {
    printf("Usage:\n");
    printf("  ./poisson2d --case <caseDir>\n\n");
}
static void message(const std::string& str,const int& rank) {
    if(rank==0){
        printf("\n\n%s\n",str.c_str());
    }
}
void printConfig(const SolverConfig &cfg,const int rank){
    if (rank != 0) return;

    printf("\n========== Solver Config ==========\n");

    printf("[wall / physical]\n");
    printf("  uwall           = %d\n", cfg.uwall);
    printf("  tauw            = %.12g\n", cfg.tauw);
    printf("  delta           = %.12g\n", cfg.delta);
    printf("  gamma           = %.12g\n", cfg.gamma);
    printf("  Pr              = %.12g\n", cfg.Pr);
    printf("  St              = %.12g\n", cfg.St);

    printf("\n[velocity space]\n");
    printf("  Nvx             = %d\n", cfg.Nvx);
    printf("  Nvy             = %d\n", cfg.Nvy);
    printf("  Nvz             = %d\n", cfg.Nvz);
    printf("  Nv              = %d\n", cfg.Nv);
    printf("  Lvx             = %.12g\n", cfg.Lvx);
    printf("  Lvy             = %.12g\n", cfg.Lvy);
    printf("  Lvz             = %.12g\n", cfg.Lvz);

    printf("\n[iteration]\n");
    printf("  max_iter        = %d\n", cfg.max_iter);
    printf("  tol             = %.12g\n", cfg.tol);
    printf("  print_interval  = %d\n", cfg.print_interval);
    printf("  check_interval  = %d\n", cfg.check_interval);

    printf("===================================\n\n");
}
// int main(int argc, char** argv) {
//     testMesh(argc,argv);
//     return 1;
// }
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double t0,t1;

    message("[read config]",rank);
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) {
            config_path = utils::join_path(argv[i + 1], "config.ini");
            break;
        }
    }
    if (config_path.empty()) {
        if (rank == 0) {
            usage();
        }
        MPI_Finalize();
        return 1;
    }

    Config cfg;
    if (!cfg.load(config_path)) {
        if (rank == 0) {
            printf("Failed to read config: %s\n\n", config_path.c_str());
        }
        MPI_Finalize();
        return 1;
    }
    std::string base_dir = utils::dirname(config_path);
    std::string mesh_file = cfg.get_string("case", "mesh_file", "mesh");
    std::string output_dir = cfg.get_string("case", "output_dir", "output");

    if (mesh_file.empty()) {
        if (rank == 0) {
            printf("Config missing case.mesh_file\n\n");
        }
        MPI_Finalize();
        return 1;
    }
    mesh_file = utils::join_path(base_dir, mesh_file);
    output_dir = utils::join_path(base_dir, output_dir);

    // read solver config
    SolverConfig scfg;
    scfg.uwall = cfg.get_int("solver", "uwall", 0);
    scfg.tauw = cfg.get_double("solver", "tauw", 0.0);
    scfg.delta = cfg.get_double("solver", "delta", 0.0);
    scfg.gamma = cfg.get_double("solver", "gamma", 5.0/3.0);
    scfg.Pr = cfg.get_double("solver", "Pr", 2.0/3.0);
    scfg.St = cfg.get_double("solver", "St", 0.0);
    scfg.Nvx = cfg.get_int("solver", "Nvx", 0);
    scfg.Nvy = cfg.get_int("solver", "Nvy", 0);
    scfg.Nvz = cfg.get_int("solver", "Nvz", 0);
    scfg.Nv = scfg.Nvx * scfg.Nvy*scfg.Nvz;
    scfg.Lvx = cfg.get_double("solver", "Lvx", 0.0);
    scfg.Lvy = cfg.get_double("solver", "Lvy", 0.0);
    scfg.Lvz = cfg.get_double("solver", "Lvz", 0.0);
    // iteration settings
    scfg.max_iter = cfg.get_int("solver", "maxIter", 2000);
    scfg.tol = cfg.get_double("solver", "tol", 1e-5);
    scfg.print_interval = cfg.get_int("solver", "printInterval", 10);
    scfg.check_interval = cfg.get_int("solver", "checkInterval", 1);
    printConfig(scfg,rank);

    // read mesh
    message("[read mesh]",rank);
    Mesh globalMesh,localMesh;
    if(rank==0){
        if(!parseFluentFile(mesh_file, globalMesh)){
            printf("Error: Failed to read Fluent mesh file: %s\n\n", mesh_file.c_str());
            MPI_Finalize();
            return 1;
        }else{
            // compute geometry
            buildCellFaces(globalMesh);// cell2face
            buildCellPoint(globalMesh);// cell2node
            calC(globalMesh);
            calV(globalMesh);
            printf("mesh file name: %s \n",mesh_file.c_str());
            printf("Total points: %lu \t Total faces: %lu \t Total cells: %lu \t\n",globalMesh.points.size(),globalMesh.faces.size(),globalMesh.cells.size());
            printf("Internal faces: %d \t Boundary faces: %d \t\n\n",globalMesh.nInternalFaces,globalMesh.nBoundaryFaces);
        }
    }
    message("[partition and distribute]",rank);
    std::string err;
    partition_and_distribute(globalMesh,localMesh,MPI_COMM_WORLD,err);
    printf("rank:%d \t local Mesh nOwned:%d \t nCells:%d \n",rank,localMesh.nOwned,localMesh.nCells);
    MPI_Barrier(MPI_COMM_WORLD);


    printf("rank %d: interior=%zu boundary=%zu owned=%d ghost=%d\n",
        rank,
        localMesh.interiorCells.size(),
        localMesh.boundaryCells.size(),
        localMesh.nOwned,
        localMesh.nCells - localMesh.nOwned);

    // message("mesh deform",rank);
    // std::vector<int> boundaryPts;
    // collectAllBoundaryPoints(globalMesh, boundaryPts);
    
    // std::vector<BoundaryNodeDisplacement> bdisp;
    // bdisp.reserve(boundaryPts.size());
    
    // double scale = 1.2;
    // for (int pid : boundaryPts)
    // {
    //     const auto& point = globalMesh.points[pid];
    //     vector d(0.0, (scale-1.0)*point.y, 0.0);
    //     bdisp.push_back({pid, d});
    // }
    
    // deformMeshSpring(globalMesh,localMesh, bdisp, MPI_COMM_WORLD, 500, 1e-10, 1.8);
    // MPI_Barrier(MPI_COMM_WORLD);

    // used for halo exchange

 
    // 修改关于邻居的定义(边界面的邻居不是-1,而是边界面对应的单元id)
    for (int bf = 0; bf < localMesh.nBoundaryFaces; ++bf) {
        int faceI = localMesh.nInternalFaces + bf;
        localMesh.faces[faceI].neigh = localMesh.nCells + bf; // boundary pseudo cell
    }


    double eps = 1e-4;
    // verifySensitivityByFiniteDifference(globalMesh,localMesh,scfg,MPI_COMM_WORLD,eps);


    // dvm 
    message("[DVM solve]",rank);
    dvmSolver dvm(localMesh,scfg,MPI_COMM_WORLD);
    
    t0 = MPI_Wtime();
    for(int iter=1; iter<scfg.max_iter; iter++){
        dvm.step(iter);
        if(dvm.res_ux < scfg.tol && dvm.res_uy < scfg.tol && dvm.res_rho < scfg.tol) {
            if (rank == 0) {
                printf("DVM converged at iteration %d\n\n", iter);
            }
            break;
        }
    }
    t1 = MPI_Wtime();
    
    report_stage_time("dvm", t1 - t0, MPI_COMM_WORLD);
    dvm.reportProfile();

    // Adjoint solver
    // message("[Adjoint solve]",rank);
    // dvm.initialAdj();
    // for(int iter=1;iter<scfg.max_iter;iter++){
    //     dvm.stepAdj(iter);
    //     if(dvm.res_aux < scfg.tol && dvm.res_auy < scfg.tol && dvm.res_arho < scfg.tol) {
    //         if (rank == 0) {
    //             printf("Adjoint DVM converged at iteration %d\n\n", iter);
    //         }
    //         break;
    //     }
    // }

    // compute sensitivity
    // message("[boundary sensitivity]",rank);
    // XForceFunctional obj;

    // std::vector<FaceGeomGrad> faceGrad;
    // BoundarySensitivityAssembler::assembleFaceGradients(dvm, obj, faceGrad);
    
    // std::vector<NodeGrad> nodeGrad;
    // BoundarySensitivityAssembler::accumulateNodeGradients(localMesh, faceGrad, nodeGrad);


    message("[output macro]",rank);
    int Nmacro = 10;
    write_tecplot(utils::join_path(output_dir, "macro"),globalMesh,localMesh,dvm.macro,Nmacro,MPI_COMM_WORLD);

    // message("[output adjoint macro]",rank);
    // int Namacro = 6;
    // write_tecplot(utils::join_path(output_dir, "amacro"),globalMesh,localMesh,dvm.amacro,Namacro,MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
