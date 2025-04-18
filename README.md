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
sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 -m venv env
source env/bin/activate
pip install torch numpy
```
### LINUX SETUP
```bash
mkdir -p vendor && cd vendor
git clone https://github.com/ddiakopoulos/libnyquist.git
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
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
## USAGE
Generate test cases:
```bash
python tests/create_tests.py
```
## Run the denoiser:
### Linux Usage
```bash
cd build
./Denoiser.cpp input_dir/ output_dir/ 4096 10
```
 Example:
```bash
cd build
./Denoiser.cpp ../dataset/noisy/ ../output
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