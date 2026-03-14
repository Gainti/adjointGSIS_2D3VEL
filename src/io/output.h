#pragma once

#include "mesh.h"

#include <mpi.h>
#include <string>
#include <vector>

#include "TECIO.h"

// global
struct TecplotOutputData
{
    int numNodes = 0;
    int numElements = 0;
    int Nvar = 0;

    // nodal data
    std::vector<double> x_coords;     // size = numNodes
    std::vector<double> y_coords;     // size = numNodes

    // cell-centered data
    std::vector<double>   part;      // size = numElements
    // BLOCK layout for Tecplot:
    // data[j * numElements + e] = variable j on element e
    std::vector<double> data;    // size = numElements * Nvar

    // FE connectivity, 1-based for Tecplot
    std::vector<int> connectivity;    // size = 4 * numElements
};

void write_residual_csv(const std::string& filename,
    const std::vector<double>& hist,
    MPI_Comm comm);

bool build_tecplot_output_data(
    const Mesh& globalMesh,
    const Mesh& localMesh,
    const std::vector<double>& u,
    int Nvar,
    MPI_Comm comm,
    TecplotOutputData& out);

bool write_tecplot(
    const std::string& filename,
    const TecplotOutputData& data);

bool write_tecplot(const std::string& filename,
    const Mesh& globalMesh,  
    const Mesh& localMesh, 
    const std::vector<double>& u,
    const int Nvar,
    MPI_Comm comm);