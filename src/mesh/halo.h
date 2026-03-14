#pragma once
#include "mesh.h"
#include "mpi.h"

bool buildHaloPlan(Mesh& local, const std::vector<int>& ghost_owner_ranks, MPI_Comm comm);


// ncomp: ncomp each cell
void haloExchangeCellData(
    const Mesh& local,
    double* data,      // length = nCells * ncomp
    int ncomp,
    MPI_Comm comm);