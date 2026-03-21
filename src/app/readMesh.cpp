#include "app.h"

#include "computeGeometry.h"
#include "partition.h"

bool readMesh(const std::string& mesh_file,
    Mesh& globalMesh,
    Mesh& localMesh,
    MPI_Comm comm,
    int rank) {
    if (rank == 0) {
        if (!parseFluentFile(mesh_file, globalMesh)) {
            printf("Error: Failed to read Fluent mesh file: %s\n\n", mesh_file.c_str());
            return false;
        }
        buildCellFaces(globalMesh);
        buildCellPoint(globalMesh);
        calC(globalMesh);
        calV(globalMesh);

        printf("mesh file name: %s\n", mesh_file.c_str());
        printf("Total points: %lu\tTotal faces: %lu\tTotal cells: %lu\n",
            globalMesh.points.size(),
            globalMesh.faces.size(),
            globalMesh.cells.size());
        printf("Internal faces: %d\tBoundary faces: %d\n\n",
            globalMesh.nInternalFaces,
            globalMesh.nBoundaryFaces);
    }

    if(rank==0){
        printf("Partitioning and distributing mesh...\n");
    }

    std::string err;
    partition_and_distribute(globalMesh, localMesh, comm, err);

    if (!err.empty()) {
        if (rank == 0) {
            printf("Partition error: %s\n", err.c_str());
        }
        return false;
    }

    printf("rank:%d\tlocal Mesh nOwned:%d\tnCells:%d\n",
    rank, localMesh.nOwned, localMesh.nCells);

    MPI_Barrier(comm);
    return true;
}
bool buildVelocitySpace(AppContext& ctx) {
    if (!ctx.vel.build(ctx.cfg)) {
        if (ctx.rank == 0) {
            printf("Error: failed to build velocity space\n");
        }
        return false;
    }
    if(ctx.rank == 0) {
        printf("Velocity space built: Nv=%d\n", ctx.cfg.Nv);
    }
    return true;
}