#ifndef LSQ_VALIDATION_H
#define LSQ_VALIDATION_H

#include "mesh.h"
#include "mpi.h"

void validateLeastSquaresLinearField(
    const Mesh& mesh,
    MPI_Comm comm,
    double ax,
    double ay,
    double b);

#endif // LSQ_VALIDATION_H
