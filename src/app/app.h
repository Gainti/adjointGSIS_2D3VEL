#pragma once

#include "defs.h"
#include "mesh.h"
#include "velocitySpace.h"
#include "dvmSolver.h"
#include "adjointDVM.h"
#include "mpi.h"

struct CaseInfo {
    std::string case_dir;
    std::string config_path;
    std::string mesh_file;
    std::string output_dir;
};

struct AppContext {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 0;
    CaseInfo case_info;
    SolverConfig cfg;
    Mesh globalMesh;
    Mesh localMesh;
    VelocitySpace vel;

    dvmSolver*  dvm = nullptr;
    adjointDVM* adj = nullptr;
};

bool readCaseConfig(int argc, char** argv,
    CaseInfo& case_info,
    SolverConfig& scfg,
    int rank);

void printConfig(const SolverConfig& cfg, int rank);

bool readMesh(const std::string& mesh_file,
    Mesh& globalMesh,
    Mesh& localMesh,
    MPI_Comm comm,
    int rank);

bool buildVelocitySpace(AppContext& ctx);

bool initializeApp(int argc, char** argv, AppContext& ctx);

bool runValidationTasks(AppContext& ctx);
bool runForwardSolve(AppContext& ctx);
bool runAdjointSolve(AppContext& ctx);
bool writeOutputs(AppContext& ctx);

bool runApp(AppContext& ctx);