cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(custom_ops)

# Enable better optimization diagnostics
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Torch path
set(Torch_DIR "/home/torfinhell/Denoiser.cpp/main_app/libtorch/share/cmake/Torch/")

# Find dependencies
find_package(Torch REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# Add executable
file(GLOB SOURCES "sources/*.cpp")
add_executable(example_app ${SOURCES})

# Modern target-based configuration
target_compile_features(example_app PRIVATE cxx_std_17)

# Add definitions for Eigen threading
target_compile_definitions(example_app PRIVATE
    EIGEN_USE_THREADS
    EIGEN_HAS_CXX11_ATOMIC
)

set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example" FORCE)
add_subdirectory("libnyquist")

# Include directories
target_include_directories(example_app PRIVATE 
    ${EIGEN3_INCLUDE_DIRS}  
    ${TORCH_INCLUDE_DIRS}  
    "libnyquist/include"
)

# Link libraries
target_link_libraries(example_app PRIVATE
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    Threads::Threads
    libnyquist::libnyquist
)

# Set compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -DEIGEN_FAST_MATH=0 -O0") # disables optimization in Eigen and certain optimizations while debugging
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -ffast-math -DNDEBUG -funroll-loops -flto")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # extra warnings

# Set position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # for shared libraries

# Linker options
target_link_options(example_app PRIVATE
    -flto
    -fuse-ld=gold
)

# Define a macro NDEBUG for Eigen3 release builds
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG") # asserts are ignored