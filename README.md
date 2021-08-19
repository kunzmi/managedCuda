It all started as a hobby project to easily access CUDA from C# - at that time CUDA was available in version 3. Now a bit more than 10 years later, while other CUDA wrappers for .net appeared and mostly disappeared, managedCuda is still alive and is updated by me to the latest versions of CUDA. And till today it is only a hobby project maintained in my spare time. Whereas managedCuda found its way into several commercial products, the donation button lacks some success: about 30€ in total for the work of 10 years :)  
I will thus change the license for upcoming releases, why this is likely the last update that I'll release under the terms of the LGPL license. From CUDA 12 on, I'll switch to a dual-license GPL / commercial license so that I can ask for a small contribution to keep this library up to date.

[![Flattr this git repo](http://api.flattr.com/button/flattr-badge-large.png)](https://flattr.com/submit/auto?user_id=kunzmi&url=https://github.com/kunzmi/managedCuda&title=managedCuda&language=&tags=github&category=software)
or
[![Support via PayPal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.me/kunzmi/)


# managedCuda
ManagedCUDA aims an easy integration of NVidia's CUDA in .net applications written in C#, Visual Basic or any other .net language.

For this it includes:
- A complete wrapper for the  CUDA Driver API, version 11.4 (a 1:1 representation of cuda.h in C#) 
- Based on this, wrapper classes for CUDA context, kernel, device variable, etc. 
- Wrapper for graphics interop with DirectX and OpenGL, respectively SlimDX and OpenTK 
- CUDA vector types like int2, float3 etc. with ToString() methods and operators (+, –, *, /) 
- Define your own types: CudaDeviceVariable accepts any user defined type if it is a value type, i.e. a struct in C# 
- Includes CUDA libraries: CUFFT, CURAND, CUSPARSE, CUBLAS, CUSOLVER, NPP, NvJPEG and NVRTC
- Compatibility for .net Framework and .net Core >3.x.
- Native Linux support for .net Core 3.x: Automatically switches the native library names.
- Access device memory directly per element using [] operator:
```
CudaDeviceVariable<float> devVar = new CudaDeviceVariable<float>(64);
devVar[0] = 1.0f;
devVar[1] = 2.0f;
float hostVar1 = devVar[0];
float hostVar2 = devVar[1];
```
- Implicit converter operators: Allocate and initialize device or host arrays in only one line of code: 
```
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
```
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
