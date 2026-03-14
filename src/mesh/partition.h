#pragma once

#include "mesh.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include "defs.h"
#include <unordered_map>


struct FacePack {
    int gFace;
    int n1, n2;     // global node ids
    int owner;      // global cell id
    int neigh;      // global cell id, or -1 if boundary
    int bc;         // BCType as int
};

struct PointPack {
    int gPoint;
    double x,y,z;
};

struct LocalPack {
	std::vector<FacePack> faces;
	std::vector<PointPack> pts;
	std::vector<int> cell_gids;
	std::vector<int> ghost_owner_ranks;  // cell_gids.size() - owned_count
	int owned_count = 0;
};

static inline int ownerRank_block(int gCell, int nCells, int nRanks) {
    // block partition: contiguous ranges
    long long begin = (long long)gCell * nRanks / nCells;
    return (int)begin;
}

bool partition_and_distribute(
	Mesh& mesh, 
	Mesh& local,
	MPI_Comm comm, std::string& err
);