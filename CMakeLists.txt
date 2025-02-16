cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_ops)

# Enable exporting compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the Torch_DIR to the path where TorchConfig.cmake is located
set(Torch_DIR "/home/torfinhell/Denoiser.cpp/venv/lib/python3.12/site-packages/torch/share/cmake/Torch")

# Find the Torch package
find_package(Torch REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Gather all source files in the main_app directory
file(GLOB SOURCES "main_app/*.cpp")

# Add the executable
add_executable(example-app ${SOURCES})

# Set the C++ standard to C++17
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# Specify include directories for the target after it has been created
target_include_directories(example-app PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(example-app PRIVATE ${TORCH_INCLUDE_DIRS})

# Link the Torch libraries
target_link_libraries(example-app "${TORCH_LIBRARIES}")
