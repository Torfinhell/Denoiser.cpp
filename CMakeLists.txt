cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_ops)

# Enable better optimization diagnostics
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Torch path
set(Torch_DIR "/home/torfinhell/Denoiser.cpp/venv/lib/python3.12/site-packages/torch/share/cmake/Torch")

# Find dependencies
find_package(Torch REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Enhanced OpenMP setup
find_package(OpenMP REQUIRED)  # Make OpenMP required
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Add executable
file(GLOB SOURCES "main_app/*.cpp")
add_executable(example-app ${SOURCES})

# Modern target-based configuration
target_compile_features(example-app PRIVATE cxx_std_17)

# Include directories
target_include_directories(example-app PRIVATE 
    ${EIGEN3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(example-app PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)

# Aggressive optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(example-app PRIVATE
        -Ofast
        -ffast-math
        -march=native
        -DNDEBUG
        -fopenmp-simd  # Enable SIMD parallelization
        -funroll-loops
        -flto  # Link-time optimization
    )
    
    target_link_options(example-app PRIVATE
        -flto
        -fuse-ld=gold  # Faster linker
    )
endif()

# For Clang specifically
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     target_compile_options(example-app PRIVATE
#         -Rpass=.*
#         -Rpass-missed=.*
#         -Rpass-analysis=.*
#     )
# endif()