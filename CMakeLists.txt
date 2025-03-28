cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(custom_ops)

# Enable better optimization diagnostics
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Torch path
set(Torch_DIR "/home/torfinhell/Denoiser.cpp/libtorch/share/cmake/Torch/")


# Find dependencies
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# Add executable
file(GLOB SOURCES "main_app/*.cpp")
add_executable(example_app ${SOURCES})

# Modern target-based configuration
target_compile_features(example_app PRIVATE cxx_std_17)

# Add definitions for Eigen threading
target_compile_definitions(example_app PRIVATE
    EIGEN_USE_THREADS
    EIGEN_HAS_CXX11_ATOMIC
)

# Include directories
target_include_directories(example_app PRIVATE 
    ${EIGEN3_INCLUDE_DIRS}  
    ${TORCH_INCLUDE_DIRS}  
)

# Link libraries
target_link_libraries(example_app PRIVATE
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    Threads::Threads
)

# Aggressive optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(example_app PRIVATE
        -Ofast
        -ffast-math
        -march=native
        -DNDEBUG
        -funroll-loops
        -flto  # Link-time optimization
    )
    
    target_link_options(example_app PRIVATE
        -flto
        -fuse-ld=gold  # Faster linker
    )
endif()
