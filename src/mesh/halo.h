#pragma once
#include "mesh.h"
#include "mpi.h"
#include <vector>

bool buildHaloPlan(Mesh& local, const std::vector<int>& ghost_owner_ranks, MPI_Comm comm);
void buildInteriorBoundaryCells(Mesh& local);

struct HaloWorkspace
{
    int ncomp = 0;

    std::vector<int> send_counts_comp;
    std::vector<int> recv_counts_comp;
    std::vector<int> send_displs_comp;
    std::vector<int> recv_displs_comp;

    std::vector<double> send_buffer;
    std::vector<double> recv_buffer;

    std::vector<MPI_Request> requests; // [0,nnei) recv, [nnei,2*nnei) send
};

struct HaloExchangeStats
{
    double t_post_recv = 0.0;
    double t_pack      = 0.0;
    double t_post_send = 0.0;
    double t_wait      = 0.0;
    double t_unpack    = 0.0;
    double t_total     = 0.0;
};

void initHaloWorkspace(
    const Mesh& local,
    HaloWorkspace& ws,
    int ncomp);

void haloExchangeBegin(
    const Mesh& local,
    HaloWorkspace& ws,
    const double* data,
    int ncomp,
    MPI_Comm comm,
    HaloExchangeStats* stats = nullptr);

void haloExchangeEnd(
    const Mesh& local,
    HaloWorkspace& ws,
    double* data,
    int ncomp,
    MPI_Comm comm,
    HaloExchangeStats* stats = nullptr);

void haloExchange(
    const Mesh& local,
    HaloWorkspace& ws,
    double* data,
    int ncomp,
    MPI_Comm comm,
    HaloExchangeStats* stats = nullptr);


void haloExchangeCellData(
    const Mesh& local,
    double* data,
    int ncomp,
    MPI_Comm comm);