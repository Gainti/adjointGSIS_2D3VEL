cmake_minimum_required(VERSION 3.15)
project(AdjointGSIS_2D3VEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

find_package(MPI REQUIRED)

# ============================================================================
# Third-party libraries
# ============================================================================

set(METIS_INCLUDE_DIR "/home/emc2/Software/lib/metis_lib/include")
set(METIS_LIBRARY "/home/emc2/Software/lib/metis_lib/lib/libmetis.so")

set(GK_INCLUDE_DIR "/home/emc2/Software/lib/gk_lib/include")
set(GK_LIBRARY "/home/emc2/Software/lib/gk_lib/lib/libGKlib.a")

# tecio
include_directories(./tecio/)
link_directories(./tecio/)

# ============================================================================
# Source files
# ============================================================================

set(SRC_APP
    src/app/readConfig.cpp
    src/app/readMesh.cpp
    src/app/app.cpp
)

set(SRC_CORE
    src/core/config.cpp
    src/core/utils.cpp
    src/core/object.cpp
)

set(SRC_MESH
    src/mesh/mesh.cpp
    src/mesh/partition.cpp
    src/mesh/halo.cpp
    src/mesh/computeGeometry.cpp
)

set(SRC_DVM
    src/dvm/dvmSolver.cpp
    src/dvm/velocitySpace.cpp
    src/dvm/adjointDVM.cpp
)

set(SRC_IO
    src/io/output.cpp
)

set(SRC_SENSITIVITY
    src/sensitivity/boundary_sensitivity.cpp
)

set(SRC_SHAPE
    src/shape/meshDeform.cpp
)

set(SRC_OPTIMIZATION
    # src/optimization/optimizer.cpp
    # src/optimization/optimize_driver.cpp
)

set(SRC_TEST
    src/test/adjoint_validation.cpp
    src/test/lsq_validation.cpp
    # src/test/geom_chain_validation.cpp
)

set(SRC_MAIN
    src/main.cpp
)

# ============================================================================
# Target
# ============================================================================

add_executable(solver)

target_sources(solver PRIVATE
    ${SRC_MAIN}
    ${SRC_CORE}
    ${SRC_MESH}
    ${SRC_DVM}
    ${SRC_IO}
    ${SRC_SENSITIVITY}
    ${SRC_TEST}
    ${SRC_SHAPE}
    ${SRC_OPTIMIZATION}
    ${SRC_APP}
)

target_include_directories(solver PRIVATE
    src
    src/core
    src/io
    src/mesh
    src/dvm
    src/sensitivity
    src/test
    src/shape
    src/optimization
    src/app
    ${METIS_INCLUDE_DIR}
    ${GK_INCLUDE_DIR}
    ${TECIO_INCLUDE_DIR}
)


target_link_libraries(solver PUBLIC 
    tecio
    MPI::MPI_CXX
    pthread
    ${METIS_LIBRARY}
    ${GK_LIBRARY}
)
