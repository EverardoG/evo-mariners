cmake_minimum_required(VERSION 3.14)  # 3.14+ recommended for better find_package behavior

project(evo_mariners CXX)

# Ensure CMake can find pagmo inside conda
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

# Pagmo requires this call to find the imported target
find_package(Pagmo REQUIRED)

# Output binaries to top-level bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Define your executable
add_executable(getting_started getting_started.cpp)

# Link to Pagmo
target_link_libraries(getting_started Pagmo::pagmo)

# Optional: Embed rpath so the binary can find Pagmo at runtime
set_target_properties(getting_started PROPERTIES
    INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
)
