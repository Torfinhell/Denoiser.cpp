cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(custom_ops)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR "vendor/libtorch/share/cmake/Torch/")

find_package(Torch REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

file(GLOB ALL_SOURCES "sources/*.cpp")
set(SOURCE_FILES "")

foreach(SOURCE ${ALL_SOURCES})
    string(REGEX MATCH "main\\.cpp$|main_test\\.cpp$" MATCH_RESULT "${SOURCE}")
    if(NOT MATCH_RESULT)
        list(APPEND SOURCE_FILES ${SOURCE})
    endif()
endforeach()

file(GLOB SOURCES "sources/*.cpp")
add_executable(Denoiser.cpp sources/main.cpp ${SOURCE_FILES})

target_compile_features(Denoiser.cpp PRIVATE cxx_std_17)

target_compile_definitions(Denoiser.cpp PRIVATE
    EIGEN_USE_THREADS
    EIGEN_HAS_CXX11_ATOMIC
)

set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example" FORCE)
add_subdirectory("vendor/libnyquist")

target_include_directories(Denoiser.cpp PRIVATE 
    ${EIGEN3_INCLUDE_DIRS}  
    ${TORCH_INCLUDE_DIRS}  
    "vendor/libnyquist/include"
)

target_link_libraries(Denoiser.cpp PRIVATE
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    Threads::Threads
    libnyquist::libnyquist
)

set(CMAKE_CXX_FLAGS_DEBUG "-g -DEIGEN_FAST_MATH=0 -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -ffast-math -DNDEBUG -funroll-loops -flto")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

target_link_options(Denoiser.cpp PRIVATE
    -flto
    -fuse-ld=gold
)

enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(Denoiser.cpp.test sources/main_test.cpp ${TEST_SOURCES} ${SOURCE_FILES})
target_link_libraries(Denoiser.cpp.test PRIVATE
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    Threads::Threads
    libnyquist::libnyquist
)
