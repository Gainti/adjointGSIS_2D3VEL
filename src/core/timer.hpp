#pragma once

#include "mpi.h"

static void report_stage_time(const char* name, double t_local, MPI_Comm comm) {
    int rank=0, size=1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double t_min=0.0, t_max=0.0, t_sum=0.0;
    MPI_Reduce(&t_local, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_local, &t_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0) {
        double t_avg = t_sum / (double)size;
        printf("[TIMER] %-18s  min=%9.6fs  avg=%9.6fs  max=%9.6fs\n\n",
               name, t_min, t_avg, t_max);
    }
}