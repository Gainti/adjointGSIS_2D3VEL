#pragma once

#include "mesh.h"
#include "mpi.h"

void buildCellFaces(Mesh& mesh);
void buildCellPoint(Mesh& mesh);
void adjustCell2node(Mesh& mesh);

void haloExchangeCellGeom(Mesh& local, MPI_Comm comm);

void computeGeometry(Mesh& mesh,MPI_Comm comm);

void calV(Mesh& mesh);
void calSf(Mesh& mesh);
void calEf(Mesh& mesh);
void calC(Mesh& mesh);
void calCf(Mesh& mesh);
void calGC(Mesh& mesh);
void caldelta(Mesh& mesh);