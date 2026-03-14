#pragma once

#include "mpi.h"
#include "mesh.h"
#include <string>
#include "partition.h"
#include "assert.h"
#include "string.h"
#include <fstream>
#include <sstream>
#include <cstdio>


bool checkMesh(Mesh mesh,MPI_Comm comm){
    // check owned and ghost
    assert(mesh.nOwned>=0 && mesh.nOwned<=mesh.nCells);
    assert(mesh.l2g_cell.size()== mesh.nCells);
    assert(mesh.g2l_cell.size()== mesh.nCells);
    assert(mesh.l2g_point.size()== mesh.points.size());
    assert(mesh.g2l_point.size()== mesh.points.size());

    for (int lc = 0; lc < mesh.nCells; ++lc) {
        int gc = mesh.l2g_cell[lc];
        auto it = mesh.g2l_cell.find(gc);
        assert(it != mesh.g2l_cell.end());
        assert(it->second == lc);
    }

    int nLocalInternalFaces=0,nHaloInterfaceFaces=0,nPhysicalBoundaryFaces=0;
    for(auto& face:mesh.faces){
        int n1=face.n1;
        int n2=face.n2;
        int owner = face.owner;
        int neigh = face.neigh;
        assert(n1>=0 && n1<mesh.points.size());
        assert(n2>=0 && n2<mesh.points.size());
        assert(owner>=0 && owner<mesh.nCells);

        if(neigh>=mesh.nCells || neigh==-1){
            nPhysicalBoundaryFaces++;
        }else{
            if(owner>=mesh.nOwned || neigh>=mesh.nOwned){
                nHaloInterfaceFaces++;
            }else{
                nLocalInternalFaces++;
            }
        }
    }
    assert(nPhysicalBoundaryFaces==mesh.nBoundaryFaces);
    assert(nLocalInternalFaces+nHaloInterfaceFaces==mesh.nInternalFaces);
    // halo plan
    for (int lc : mesh.halo.send_cells) {
        assert(lc >= 0 && lc < mesh.nOwned);
    }
    for (int lc : mesh.halo.recv_cells) {
        assert(lc >= mesh.nOwned && lc < mesh.nCells);
    }

    return true;
}

bool testMesh(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // read mesh
    const std::string mesh_file("./cases/test/mesh/testMesh_4.cas");
    Mesh GlobalMesh;
    if(rank==0){
        if(!parseFluentFile(mesh_file, GlobalMesh)){
            printf("Error: Failed to read Fluent mesh file: %s\n\n", mesh_file.c_str());
            MPI_Finalize();
            return 1;
        }else{
            printf("mesh file name: %s \n",mesh_file.c_str());
            printf("Total points: %lu \t Total faces: %lu \t Total cells: %lu \t\n",GlobalMesh.points.size(),GlobalMesh.faces.size(),GlobalMesh.cells.size());
            printf("Internal faces: %d \t Boundary faces: %d \t\n\n",GlobalMesh.nInternalFaces,GlobalMesh.nBoundaryFaces);
        }
    }
    std::string err;
    Mesh mesh;
    partition_and_distribute(GlobalMesh,mesh,MPI_COMM_WORLD,err);
    printf("rank:%d \t local Mesh nOwned:%d \t nCells:%d \n",rank,mesh.nOwned,mesh.nCells);

    checkMesh(mesh,MPI_COMM_WORLD);

    // check C,V,cell2face,Cf,Sf
    char filename[256];
    std::snprintf(filename, sizeof(filename), "./cases/test/output_rank_%d.dat", rank);

    FILE* fp = std::fopen(filename, "w");
    if (fp == nullptr) {
        std::fprintf(stderr, "Rank %d: failed to open file %s\n", rank, filename);
        MPI_Finalize();
        return 1;
    }
    const auto& l2g_cell = mesh.l2g_cell;
    for(int celli=0;celli<mesh.nCells;celli++){
        const auto& cell=mesh.cells[celli];
        int cell_g = l2g_cell[celli];
        fprintf(fp,"cell_g:%d \t C:(%.3e,%.3e) \t V:%e\n",cell_g,cell.C.x,cell.C.y,cell.V);
        for(auto facei:cell.cell2face){
            int owner=l2g_cell[mesh.faces[facei].owner];
            int neigh=mesh.faces[facei].neigh;
            if(neigh>=0){
                neigh=l2g_cell[neigh];
            }
            const auto& Cf = mesh.faces[facei].Cf;
            const auto& Sf = mesh.faces[facei].Sf;
            fprintf(fp,"\t facei:%d \t owner_g:%d neigh_g:%d \t Cf:(%.3e,%.3e) \t Sf:(%.3e,%.3e) \n",facei,owner,neigh,Cf.x,Cf.y,Sf.x,Sf.y);
        }
        fprintf(fp,"\n");
    }
    std::fclose(fp);

    MPI_Finalize();
    return true;
}