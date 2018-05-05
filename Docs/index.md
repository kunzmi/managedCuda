# ManagedCuda.NETStandard

[![Build Status](https://travis-ci.org/surban/managedCuda.svg?branch=master)](https://travis-ci.org/surban/managedCuda)

Donate a beer to help the original author keep managedCuda up to date :)
[![Flattr this git repo](http://api.flattr.com/button/flattr-badge-large.png)](https://flattr.com/submit/auto?user_id=kunzmi&url=https://github.com/kunzmi/managedCuda&title=managedCuda&language=&tags=github&category=software)
or
[![Support via PayPal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.me/kunzmi/)

This is a port of [ManagedCuda](https://kunzmi.github.io/managedCuda/) to .NET Standard 2.0.
It has been tested on Linux and Microsoft Windows.

ManagedCUDA aims an easy integration of NVidia's CUDA in .NET applications written in C#, F#, Visual Basic or any other .NET language.

For this it includes:
- A complete wrapper for the  CUDA Driver API, version 9.1 (a 1:1 representation of cuda.h in C#) 
- Based on this, wrapper classes for CUDA context, kernel, device variable, etc. 
- Wrapper for graphics interop with DirectX and OpenGL, respectively SlimDX and OpenTK 
- CUDA vector types like int2, float3 etc. with ToString() methods and operators (+, –, *, /) 
- Define your own types: CudaDeviceVariable accepts any user defined type if it is a value type, i.e. a struct in C# 
- Includes all CUDA libraries: CUFFT, CURAND, CUSPARSE, CUBLAS, CUSOLVE, NPP, NVGRAPH, Nvml and NVRTC (not all of them are up to date yet)
- Access device memory directly per element using [] operator:
  ```csharp
  CudaDeviceVariable<float> devVar = new CudaDeviceVariable<float>(64);
  devVar[0] = 1.0f;
  devVar[1] = 2.0f;
  float hostVar1 = devVar[0];
  float hostVar2 = devVar[1];
  ```
- Implicit converter operators: Allocate and initialize device or host arrays in only one line of code: 
  ```csharp
  float3[] array_host = new float3[100];
  for (int i = 0; i < 100; i++)
  {
    array_host[i] = new float3(i, i+1, i+2);
  }
  //alloc device memory and copy data:
  CudaDeviceVariable<float3> array_device = array_host;
  //alloc host array and copy data: 
  float3[] array_host2 = array_device; 
  ```
- NPPs extension methods for CudaDeviceVariable. Add a reference to the NPP library and include the ManagedCuda.NPP.NPPsExtensions namespace: 
  ```csharp
  Random rand = new Random();
  int length = 256;
  
  //init some ramdom values
  double[] randoms = new double[length];
  for (int i = 0; i < length; i++)
  {
  	randoms[i] = rand.NextDouble();
  }
  
  //Alloc device memory
  CudaDeviceVariable<double> a = randoms;
  CudaDeviceVariable<double> b = new CudaDeviceVariable<double>(length);
  b.Set(10.0); //NPPs method
  int size = a.MeanGetBufferSize(); //NPPs method
  //Alloc temporary memory for NPPs mean method
  CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(size);
  CudaDeviceVariable<double> mean = new CudaDeviceVariable<double>(1);
  
  a.Mul(b); //NPPs method
  a.DivC(10.0); //NPPs method
  a.Mean(mean, buffer); //NPPs method
  
  //Copy data back to host
  double m = mean;
  double[] res = a;
  
  //Clean up
  mean.Dispose();
  buffer.Dispose();
  b.Dispose();
  a.Dispose();
  ```
- Compiles for .NET Standard 2.0, runs on Windows and Linux. 
- The new feature 'per thread default stream' is available as a compiler directive of the managedCuda main library: Compile the library with the option "_PerThreadDefaultStream" to enable it.

## NuGet packages

Prebuilt NuGet packages are [available on nuget.org](https://www.nuget.org/packages?q=ManagedCuda+NETStandard).

## Documentation

You can read the [full API reference documentation](api/index.md).

## Building from source

Source code is available at <https://github.com/surban/managedCuda>.

For building you need the following software:

- .NET Core SDK 2.0
- CUDA SDK

Place the required native libraries from the CUDA SDK into the `Redist` folder.

To obtain NuGet packages for a Release build, run from the root directory:

```dotnet pack -c Release```

The built packages are located in the ```Packages/Release``` directory.

See ```ManagedCUDA.targets``` for common build settings.
