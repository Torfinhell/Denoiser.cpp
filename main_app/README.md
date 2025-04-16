# Command Line Audio Denoiser C++
Implemented from [Denoiser Paper](https://arxiv.org/pdf/2006.12847) [Denoiser Implementation on gihub in Python]( https://github.com/facebookresearch/denoiser).

---

## INSTALLATION

### Prerequisites
- Python 3.6+ (for test scripts)
- C++17 compatible compiler
- CMake 3.18+
- Git


### Install Python dependencies
```bash
pip install torch numpy
```
### LINUX SETUP
```bash
mkdir -p vendor && cd vendor
git clone https://github.com/ddiakopoulos/libnyquist.git
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
### WINDOWS SETUP
```bash
mkdir vendor
cd vendor
git clone https://github.com/ddiakopoulos/libnyquist.git
$url = "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip"
Invoke-WebRequest -Uri $url -OutFile libtorch.zip
Expand-Archive -Path libtorch.zip -DestinationPath .
```
## BUILDING THE PROJECT
### LINUX BUILD
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="../vendor/libtorch" \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_COMPILER=clang \
         -DCMAKE_CXX_COMPILER=clang++
make -j$(nproc)
```
### WINDOWS BUILD
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 `
         -DCMAKE_PREFIX_PATH="$PWD\..\vendor\libtorch"
cmake --build . --config Release
```
## USAGE
Generate test cases:
```bash
python tests/create_tests.py
```
## Run the denoiser:
### Linux/Windows Usage
```bash
./build/denoiser input_dir/ output_dir/ 4096 10
```
### Command Line Arguments:
```bash
input_path      Input file/directory path
output_path     Output file/directory path
stride  Processing stride (default: 4096)
threads Number of worker threads (default: 10)
```
## LICENSE
MIT License - See LICENSE for details.