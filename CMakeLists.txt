cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_ops)

# Find the Torch package
find_package(Torch REQUIRED)

# Add the executable
add_executable(example-app main_app/main.cpp)

# Link the Torch libraries
target_link_libraries(example-app "${TORCH_LIBRARIES}")

# Set the C++ standard to C++17
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# Optional: If you want to enable CUDA support, you can uncomment the following lines
# find_package(CUDA REQUIRED)
# target_link_libraries(example-app "${CUDA_LIBRARIES}")
