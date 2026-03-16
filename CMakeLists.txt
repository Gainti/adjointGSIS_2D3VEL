cmake_minimum_required(VERSION 3.15)
project(AdjointGSIS_2D3VEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

find_package(MPI REQUIRED)

# set(METIS_INCLUDE_DIR "/usr/include")
# set(METIS_LIBRARY "/usr/lib/x86_64-linux-gnu/libmetis.so")
# set(METIS_LIBRARY "/usr/lib/x86_64-linux-gnu/libmetis.so.5.1.0")

set(METIS_INCLUDE_DIR "/home/emc2/Software/lib/metis_lib/include")
set(METIS_LIBRARY "/home/emc2/Software/lib/metis_lib/lib/libmetis.so")

set(GK_INCLUDE_DIR "/home/emc2/Software/lib/gk_lib/include")
set(GK_LIBRARY "/home/emc2/Software/lib/gk_lib/lib/libGKlib.a")

# tecio
include_directories(./tecio/)
link_directories(./tecio/)
# teciompi
# include_directories(./teciompi/)
# link_directories(./teciompi/)

add_executable(solver
    src/main.cpp
    src/core/config.cpp
    src/mesh/mesh.cpp
    src/core/utils.cpp
    src/dvm/dvmSolver.cpp
    src/io/output.cpp
    src/mesh/partition.cpp
    src/mesh/halo.cpp
    src/mesh/computeGeometry.cpp
    src/mesh/meshDeform.cpp
    src/dvm/adjointDVM.cpp
    src/dvm/boundary_sensitivity.cpp
    src/test/geom_chain_validation.cpp
)

target_include_directories(solver PUBLIC 
    src 
    src/core 
    src/io 
    src/mesh 
    src/dvm
    src/test
    ${METIS_INCLUDE_DIR}
    ${GK_INCLUDE_DIR}
)
target_link_libraries(solver PUBLIC 
    tecio
    MPI::MPI_CXX
    pthread
    ${METIS_LIBRARY}
    ${GK_LIBRARY}
)
