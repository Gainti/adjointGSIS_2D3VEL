#include "mpi.h"
#include "app.h"


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    AppContext ctx;
    if(!initializeApp(argc, argv, ctx)) {
        MPI_Finalize();
        return 1;
    }

    if(!runApp(ctx)){
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
