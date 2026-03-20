#include "lsq_validation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
inline double evalLinear(double x, double y, double ax, double ay, double b) {
    return ax * x + ay * y + b;
}
}

void validateLeastSquaresLinearField(
    const Mesh& mesh,
    MPI_Comm comm,
    double ax,
    double ay,
    double b)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    double max_err_x_local = 0.0;
    double max_err_y_local = 0.0;
    double l2_err_sum_local = 0.0;
    long long n_local = 0;

    for (int cellI = 0; cellI < mesh.nOwned; ++cellI) {
        const auto& cell = mesh.cells[cellI];

        const double phi_p = evalLinear(cell.C.x, cell.C.y, ax, ay, b);
        double b0 = 0.0;
        double b1 = 0.0;

        for (int faceI : cell.cell2face) {
            const auto& face = mesh.faces[faceI];
            const int owner = face.owner;
            const int neigh = face.neigh;

            vector dC;
            double phi_nb = 0.0;

            if (cellI == owner) {
                if (neigh < mesh.nCells) {
                    const auto& Cnb = mesh.cells[neigh].C;
                    dC = Cnb - cell.C;
                    phi_nb = evalLinear(Cnb.x, Cnb.y, ax, ay, b);
                } else {
                    // Boundary pseudo-neighbor: use face center for consistency with invA build.
                    dC = face.Cf - cell.C;
                    phi_nb = evalLinear(face.Cf.x, face.Cf.y, ax, ay, b);
                }
            } else {
                const auto& Cnb = mesh.cells[owner].C;
                dC = Cnb - cell.C;
                phi_nb = evalLinear(Cnb.x, Cnb.y, ax, ay, b);
            }

            const double dphi = phi_nb - phi_p;
            b0 += dC.x * dphi;
            b1 += dC.y * dphi;
        }

        const double gx = cell.invA[0] * b0 + cell.invA[1] * b1;
        const double gy = cell.invA[1] * b0 + cell.invA[2] * b1;

        const double errx = std::abs(gx - ax);
        const double erry = std::abs(gy - ay);

        max_err_x_local = std::max(max_err_x_local, errx);
        max_err_y_local = std::max(max_err_y_local, erry);

        l2_err_sum_local += errx * errx + erry * erry;
        n_local++;
    }

    double max_err_x_global = 0.0;
    double max_err_y_global = 0.0;
    double l2_err_sum_global = 0.0;
    long long n_global = 0;

    MPI_Allreduce(&max_err_x_local, &max_err_x_global, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&max_err_y_local, &max_err_y_global, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&l2_err_sum_local, &l2_err_sum_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&n_local, &n_global, 1, MPI_LONG_LONG, MPI_SUM, comm);

    if (rank == 0) {
        const double rms = (n_global > 0) ? std::sqrt(l2_err_sum_global / static_cast<double>(2 * n_global)) : 0.0;
        std::printf("\n[LSQ linear-field validation]\n");
        std::printf("  field: phi(x,y) = %.16e * x + %.16e * y + %.16e\n", ax, ay, b);
        std::printf("  checked owned cells (global): %lld\n", n_global);
        std::printf("  max |gx-ax| = %.16e\n", max_err_x_global);
        std::printf("  max |gy-ay| = %.16e\n", max_err_y_global);
        std::printf("  RMS error   = %.16e\n\n", rms);
    }
}
