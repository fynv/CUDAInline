# CUDAInline

This is just another attempt to enable a CUDA interface within Python.

In ThrustRTC, I implemented most of the algorithms of Thrust in C++ and made Python/C#/JAVA bindings. This made things over-complicated, and writing wrapping layers is tedious.

This project is a distillation of the engine part of ThrustRTC. It provides the minimal interfaces to Python that are just enough for user to use CUDA in Python and to do more complicated stuff fully within Python, without the need of writting more C++ code.

It is possible to extend the code in Python to do all the Thrust stuff. Also possible to wrap the C++ engine to interface C#/JAVA. However, to implement everything provided by ThrustRTC, we may end up with even more code, which is against the essense of the project: keep it light-weighted, make things simple and easy.

## Installation

### Install from Source Code

Source code of CUDAInline is available at:
https://github.com/fynv/CUDAInline

The code does not actually contain any CUDA device code that need to be
prebuilt, therefore CUDA SDK is not a requirement at building time.

At build time, you will only need:
* UnQLite source code, as submodule: thirdparty/unqlite
* CMake 3.x

After cloning the repo from github and resolving the submodules, you can build it
with CMake.

```
$ mkdir build
$ cd build
$ cmake .. -DBUILD_PYTHON_BINDINGS=true -DBUILD_TESTS=true -DINCLUDE_TESTS=true
$ make
$ make install
```

You will get the library headers, binaries and examples in the "install" directory.

### Install PyCUDAInline from PyPi

Builds for Win64/Linux64 + Python 3.x are available from Pypi. If your
environment matches, you can try:

```
$ pip3 install CUDAInline
```

## Runtime Dependencies

* CUDA driver (up-to-date)
* Shared library of NVRTC 
  
  * Windows: nvrtc64\*.dll, default location: %CUDA_PATH%/bin
  * Linux: libnvrtc.so, default location: /usr/local/cuda/lib64
  
  If the library is not at the default location, you need to call:

  * set_libnvrtc_path() from C++ or 
  * CUDAInline.set_libnvrtc_path() from Python

  at run-time to specify the path of the library.

For Python
* Python 3
* cffi
* numpy
* numba (optional)

## Functionality

CUDAInlines lets you run CUDA kernels embedded in Python code just like in ThrustRTC.

The 'DVCombine' interface is provide for user to implement custom DeviceViewable classes without using C++.

The 'DVVector' class in this project is implemented using the 'DVCombine' interface, as a combination of a general DVBuffer and a DVUInt64 (for size info), in contrast to the implementation in ThrustRTC, which is wrapping of the underlying C++ class.

## License 

I've decided to license this project under ['"Anti 996" License'](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Basically, you can use the code any way you like unless you are working for a 996 company.

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)



