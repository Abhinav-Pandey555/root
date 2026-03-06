# Exercise 1 - Building ROOT from Source

## Build Configuration
- OS: Ubuntu 22.04 WSL2
- CPU: AMD Ryzen 7 5800HS
- RAM: 7.5 GB
- ROOT Version: 6.39.01

## CMake Options Used
- -Dtmva-sofie=ON
- -Dtmva=ON
- -Dpymva=ON
- -Dbuiltin_protobuf=OFF

## Dependencies
- GCC: 11.4.0
- CMake: 3.22.1
- Protobuf: 3.12.4
- Python: 3.10.12
- PyTorch: 2.5.1+cu121
- TensorFlow: 2.20.0
- NumPy: 2.2.6

## Notes
- SOFIE parser successfully built (libROOTTMVASofieParser.so)
- CUDA detected through nvidia-smi but CUDA compiler not found in WSL2
- All builtin dependencies handled automatically by ROOT
- Compilation time: approximately 2 hours
