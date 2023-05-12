It all started as a hobby project to easily access CUDA from C# - at that time CUDA was available in version 3. Now more than 10 years later, managedCuda is still alive and is updated regularly by me to the latest versions of CUDA. In order to support further developments, I switched from the LGPL license to a dual-license GPLv3 / commercial license starting with managedCuda for Cuda version 12 onwards. In case you plan to use managedCuda 12 for a commercial project, please contact me by mail: managedcuda@articimaging.eu. If you use the open-source license and want to contribute to future development, you can donate me a beer here: 
[![Support via PayPal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.me/kunzmi/)

# Official nuget packages
One can find multiple packages for managedCuda on nuget, but the official packages are:
- [ManagedCuda-12](https://www.nuget.org/packages/ManagedCuda-12/) (core library without dependencies)
- [CUBLAS](https://www.nuget.org/packages/ManagedCuda-CUBLAS) (wrapper for cuBlas library, depends on ManagedCuda-12)
- [CUFFT](https://www.nuget.org/packages/ManagedCuda-CUFFT) (wrapper for cuFFT library, depends on ManagedCuda-12)
- [CURAND](https://www.nuget.org/packages/ManagedCuda-CURAND) (wrapper for cuRand library, depends on ManagedCuda-12)
- [CUSOLVE](https://www.nuget.org/packages/ManagedCuda-CUSOLVE) (wrapper for cuSolver library, depends on ManagedCuda-12)
- [CUSPARSE](https://www.nuget.org/packages/ManagedCuda-CUSPARSE) (wrapper for cuSparse library, depends on ManagedCuda-12)
- [NPP](https://www.nuget.org/packages/ManagedCuda-NPP) (wrapper for NPP library, depends on ManagedCuda-12)
- [NVJITLINK](https://www.nuget.org/packages/ManagedCuda-NVJITLINK) (wrapper for nvJitLink library, depends on ManagedCuda-12)
- [NVJPEG](https://www.nuget.org/packages/ManagedCuda-NVJPEG) (wrapper for nvjpeg library, depends on ManagedCuda-12)
- [NVRTC](https://www.nuget.org/packages/ManagedCuda-NVRTC) (wrapper for nvrtc library, depends on ManagedCuda-12)

# managedCuda
ManagedCUDA aims an easy integration of NVidia's CUDA in .net applications written in C#, Visual Basic or any other .net language.

For this it includes:
- A complete wrapper for the  CUDA Driver API, version 12.1 (a 1:1 representation of cuda.h in C#) 
- Based on this, wrapper classes for CUDA context, kernel, device variable, etc. 
- Wrapper for graphics interop with DirectX and OpenGL, respectively SlimDX and OpenTK 
- CUDA vector types like int2, float3 etc. with ToString() methods and operators (+, –, *, /) 
- Define your own types: CudaDeviceVariable accepts any user defined type if it is a value type, i.e. a struct in C# 
- Includes CUDA libraries: CUBLAS, CUFFT, CURAND, CUSOLVER, CUSPARSE, NPP, NvJPEG, NvJitLink and NVRTC
- Compatibility for .net Framework 4.8 (might be dropped in a future version) and .net Core >3.1.
- Native Linux support for .net Core >3.1: Automatically switches the native library names.
- Access device memory directly per element using [] operator:
```C#
CudaDeviceVariable<float> devVar = new CudaDeviceVariable<float>(64);
devVar[0] = 1.0f;
devVar[1] = 2.0f;
float hostVar1 = devVar[0];
float hostVar2 = devVar[1];
```
- Implicit converter operators: Allocate and initialize device or host arrays in only one line of code: 
```C#
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
```C#
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

- The new feature 'per thread default stream' is available as a compiler directive of the managedCuda main library: Compile the library with the option "_PerThreadDefaultStream" to enable it.

# Note about Cuda context
Nvidia changed the cuda context behavior in the cuda libraries (NPP, Cufft, etc.) why it is highly recommended to use a ```PrimaryContext``` instead of a ```CudaContext``` when using ManagedCUDA together with Cuda libraries. To create a ```PrimaryContext``` in ManagedCUDA, use the following lines of code:
```C#
int deviceID = 0;
PrimaryContext ctx = new PrimaryContext(deviceID);
// Set current to CPU thread, mandatory for a PrimaryContext
ctx.SetCurrent();
```

# NppStreamContext
In order to use the NppStreamContext-API of NPP, initialize a ```NppStreamContext``` like this:
```C#
CudaStream cudaStream = new CudaStream();          //optional, 
NPPNativeMethods.NPPCore.nppSetStream(cudaStream); //if not set, NPP will work on the default null-stream
NppStreamContext nppCtx = NPPNativeMethods.NPPCore.nppGetStreamContext();
```