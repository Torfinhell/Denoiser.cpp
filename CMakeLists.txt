cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_ops)

# Set the Torch_DIR to the path where TorchConfig.cmake is located
set(Torch_DIR "/home/torfinhell/Denoiser.cpp/venv/lib/python3.12/site-packages/torch/share/cmake/Torch")

# Find the Torch package
find_package(Torch REQUIRED)

# Gather all source files in the main_app directory
file(GLOB SOURCES "main_app/*.cpp")

# Add the executable
add_executable(example-app ${SOURCES})

# Link the Torch libraries
target_link_libraries(example-app "${TORCH_LIBRARIES}")

# Set the C++ standard to C++17
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# Optional: If you want to enable CUDA support, you can uncomment the following lines
# find_package(CUDA REQUIRED)
# target_link_libraries(example-app "${CUDA_LIBRARIES}")

# Optional: If you want to include the Torch headers
target_include_directories(example-app PRIVATE ${TORCH_INCLUDE_DIRS})

# Optional: If you want to enable C++14 or C++20, you can change the CXX_STANDARD
# set_property(TARGET example-app PROPERTY CXX_STANDARD 20)



