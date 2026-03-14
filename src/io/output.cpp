#include "output.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mpi_proto.h>
#include <vector>
#include <exception>
#include <iostream>

static double u_exact(double x, double y) {
	const double pi = std::acos(-1.0);
	return std::sin(pi * x) * std::sin(pi * y);
}

void write_residual_csv(const std::string& path, const std::vector<double>& hist, MPI_Comm comm) {
	int rank = 0;
	MPI_Comm_rank(comm, &rank);
	if (rank != 0) {
		return;
	}
	FILE* fp = std::fopen(path.c_str(), "w");
	if (!fp) {
		return;
	}
	std::fprintf(fp, "iter,residual\n");
	for (size_t i = 0; i < hist.size(); ++i) {
		std::fprintf(fp, "%zu,%.15e\n", i + 1, hist[i]);
	}
	std::fclose(fp);
}

bool build_tecplot_output_data(
    const Mesh& globalMesh,
    const Mesh& localMesh,
    const std::vector<double>& u,
    int Nvar,
    MPI_Comm comm,
    TecplotOutputData& out)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int nOwned = localMesh.nOwned;

    if ((int)u.size() < nOwned * Nvar) {
        if (rank == 0) {
			printf("Error: u.size() < localMesh.nOwned * Nvar\n");
        }
        return false;
    }
	if(rank==0){
		out = TecplotOutputData{};
		out.numNodes = (int)globalMesh.points.size();
		out.numElements = globalMesh.nCells;
		out.Nvar = Nvar;
		// x,y
		out.x_coords.resize(out.numNodes);
		out.y_coords.resize(out.numNodes);
		for (int i = 0; i < out.numNodes; ++i) {
			out.x_coords[i] = globalMesh.points[i].x;
			out.y_coords[i] = globalMesh.points[i].y;
		}
		// rank
		out.part.resize(globalMesh.nOwned);
		for(int celli=0;celli<globalMesh.nOwned;celli++){
			out.part[celli] = (double) globalMesh.part[celli];
		}
		// connectivity
		out.connectivity.reserve(4 * out.numElements);
		for (int c = 0; c < out.numElements; ++c) {
			const auto& cellNodes = globalMesh.cell2node[c];
	
			if (cellNodes.size() == 4) {
				out.connectivity.push_back(cellNodes[0] + 1);
				out.connectivity.push_back(cellNodes[1] + 1);
				out.connectivity.push_back(cellNodes[2] + 1);
				out.connectivity.push_back(cellNodes[3] + 1);
			}
			else if (cellNodes.size() == 3) {
				out.connectivity.push_back(cellNodes[0] + 1);
				out.connectivity.push_back(cellNodes[1] + 1);
				out.connectivity.push_back(cellNodes[2] + 1);
				out.connectivity.push_back(cellNodes[2] + 1);
			}
			else {
				printf("Error: unsupported cell node count =%ld at cell %d \n",cellNodes.size(),c);
				return false;
			}
		}
	}
    // ----------------------------------------
    // serial case
    // ----------------------------------------
    if (size == 1) {
        // udf
        out.data.resize((size_t)out.numElements * Nvar);
		for (int j = 0; j < Nvar; ++j) {
			for (int e = 0; e < out.numElements; ++e) {
				out.data[j * out.numElements + e] = u[e * Nvar + j];
			}
		}
        return true;
    }

    // ----------------------------------------
    // parallel case: gather global cell ids + udf
    // ----------------------------------------
    std::vector<int> send_gids(nOwned);
    for (int c = 0; c < nOwned; ++c) {
        send_gids[c] = localMesh.l2g_cell[c];
    }

    std::vector<double> send_u(nOwned * Nvar);
    for (int c = 0; c < nOwned; ++c) {
        for (int j = 0; j < Nvar; ++j) {
            send_u[c * Nvar + j] = u[c * Nvar + j];
        }
    }

    std::vector<int> recv_counts_cells;
    if (rank == 0) recv_counts_cells.resize(size, 0);

    MPI_Gather(&nOwned, 1, MPI_INT,
               rank == 0 ? recv_counts_cells.data() : nullptr,
               1, MPI_INT, 0, comm);

    std::vector<int> displs_cells, recv_counts_u, displs_u;
    int total_cells = 0;

    if (rank == 0) {
        displs_cells.resize(size, 0);
        recv_counts_u.resize(size, 0);
        displs_u.resize(size, 0);

        for (int r = 0; r < size; ++r) {
            displs_cells[r] = total_cells;
            total_cells += recv_counts_cells[r];
            recv_counts_u[r] = recv_counts_cells[r] * Nvar;
        }

        int offset_u = 0;
        for (int r = 0; r < size; ++r) {
            displs_u[r] = offset_u;
            offset_u += recv_counts_u[r];
        }
    }

    std::vector<int> all_gids;
    std::vector<double> all_u;

    if (rank == 0) {
        all_gids.resize(total_cells);
        all_u.resize((size_t)total_cells * Nvar);
    }

    MPI_Gatherv(send_gids.data(), nOwned, MPI_INT,
                rank == 0 ? all_gids.data() : nullptr,
                rank == 0 ? recv_counts_cells.data() : nullptr,
                rank == 0 ? displs_cells.data() : nullptr,
                MPI_INT, 0, comm);

    MPI_Gatherv(send_u.data(), nOwned * Nvar, MPI_DOUBLE,
                rank == 0 ? all_u.data() : nullptr,
                rank == 0 ? recv_counts_u.data() : nullptr,
                rank == 0 ? displs_u.data() : nullptr,
                MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        out.data.assign((size_t)out.numElements * Nvar, 0.0);

        // fill rank_field and udf_field by gathered gids
        for (int r = 0; r < size; ++r) {
            int begin = displs_cells[r];
            int count = recv_counts_cells[r];

            for (int k = 0; k < count; ++k) {
                int gCell = all_gids[begin + k];
				for (int j = 0; j < Nvar; ++j) {
					out.data[j * out.numElements + gCell] = all_u[(begin + k) * Nvar + j];
				}
            }
        }
    }

    return true;
}
bool write_tecplot(
    const std::string& filename,
    const TecplotOutputData& data)
{
    if ((int)data.x_coords.size() != data.numNodes ||
        (int)data.y_coords.size() != data.numNodes) {
		printf("Error: nodal coordinate size mismatch\n");
        return false;
    }

    if ((int)data.part.size() != data.numElements) {
		printf("Error: part size mismatch\n");
        return false;
    }

    if ((int)data.data.size() != data.numElements * data.Nvar) {
		printf("Error: field size mismatch\n");
        return false;
    }

    if ((int)data.connectivity.size() != 4 * data.numElements) {
        std::cerr << "Error: connectivity size mismatch\n";
        return false;
    }

    int fileFormat = 1;   // 0=.plt, 1=.szplt
    int fileType = 0;
    int debug = 0;
    int vIsDouble = 1;

    std::string varNames = "X Y rank";
    for (int i = 0; i < data.Nvar; ++i) {
        varNames += " UDF" + std::to_string(i);
    }

    int result = TECINI142(
        "Field Data Output",
        varNames.c_str(),
        filename.c_str(),
        ".",
        &fileFormat,
        &fileType,
        &debug,
        &vIsDouble
    );

    if (result != 0) {
        std::cerr << "Error: Unable to initialize TecIO for file "
                  << filename << std::endl;
        return false;
    }

    const char* zoneTitle = "Field Data Zone";
    int zoneType = 3;  // FEQUADRILATERAL
    int numNodes = data.numNodes;
    int numElements = data.numElements;
    int numFaces = 0;

    int iCellMax = 0, jCellMax = 0, kCellMax = 0;
    double solutionTime = 0.0;
    int strandID = 0;
    int parentZone = 0;
    int isBlock = 1;
    int numFaceConnections = 0;
    int faceNeighborMode = 0;
    int totalNumFaceNodes = 0;
    int numConnectedBoundaryFaces = 0;
    int totalNumBoundaryConnections = 0;

    int totalVars = 3 + data.Nvar;
    std::vector<int> valueLocation(totalVars, 1);
    for (int i = 2; i < totalVars; ++i) {
        valueLocation[i] = 0; // rank + udf are cell-centered
    }

    int* passiveVarList = nullptr;
    int* shareVarFromZone = nullptr;
    int shareConnectivityFromZone = 0;

    result = TECZNE142(
        zoneTitle,
        &zoneType,
        &numNodes,
        &numElements,
        &numFaces,
        &iCellMax,
        &jCellMax,
        &kCellMax,
        &solutionTime,
        &strandID,
        &parentZone,
        &isBlock,
        &numFaceConnections,
        &faceNeighborMode,
        &totalNumFaceNodes,
        &numConnectedBoundaryFaces,
        &totalNumBoundaryConnections,
        passiveVarList,
        valueLocation.data(),
        shareVarFromZone,
        &shareConnectivityFromZone
    );

    if (result != 0) {
        std::cerr << "Error: Unable to create zone" << std::endl;
        TECEND142();
        return false;
    }

    // 1) write x
    result = TECDAT142(&numNodes,
                       const_cast<double*>(data.x_coords.data()),
                       &vIsDouble);
    if (result != 0) {
        std::cerr << "Error: Unable to write X coordinates" << std::endl;
        TECEND142();
        return false;
    }

    // 2) write y
    result = TECDAT142(&numNodes,
                       const_cast<double*>(data.y_coords.data()),
                       &vIsDouble);
    if (result != 0) {
        std::cerr << "Error: Unable to write Y coordinates" << std::endl;
        TECEND142();
        return false;
    }

    // 3) write part first
    result = TECDAT142(&numElements, data.part.data(), &vIsDouble);
    if (result != 0) {
        std::cerr << "Error: Unable to write rank field" << std::endl;
        TECEND142();
        return false;
    }

    // 4) write
	int fieldsize = data.Nvar*data.numElements;
	result = TECDAT142(&fieldsize, data.data.data(), &vIsDouble);
	if (result != 0) {
		printf("Error: Unable to write field\n");
		TECEND142();
		return false;
	}


    // 5) connectivity
    result = TECNOD142(const_cast<int*>(data.connectivity.data()));
    if (result != 0) {
        std::cerr << "Error: Unable to write connectivity" << std::endl;
        TECEND142();
        return false;
    }

    result = TECEND142();
    if (result != 0) {
        std::cerr << "Error: Unable to close file" << std::endl;
        return false;
    }

    printf("Successfully wrote field data to %s\n", filename.c_str());
    return true;
}
bool write_tecplot(const std::string& filename,
    const Mesh& globalMesh,  
    const Mesh& localMesh, 
    const std::vector<double>& u,
    const int Nvar,
    MPI_Comm comm)
{
	int rank,size;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&size);

	TecplotOutputData tecout;
	if(!build_tecplot_output_data(globalMesh,localMesh,u,Nvar,comm,tecout)){
		if(rank==0){
			printf("Error: build_tecplot_output_data\n");
		}
		return false;
	}
	if(rank==0){
		return write_tecplot(filename,tecout);
	}
	return true;
}