#!/bin/bash

set -e 
cd "$(dirname "$0")"

genstub () 
{
    DLLNAME=$1
    TARGET=$2

    rm -f empty.c
    touch empty.c
    gcc-6 -shared -o ${TARGET} empty.c    
    gcc-6 -Wl,--no-as-needed -shared -o lib${DLLNAME}.so -fPIC -L. -l:${TARGET}
    rm -f ${TARGET}
    rm -f empty.c

    echo "Mapped ${DLLNAME}.dll ==> ${TARGET}"
}


# nvcuda.dll or libcuda.so.1 and nvml.dll or libnvidia-ml.so.1 are always installed by the GPU driver.
genstub nvcuda libcuda.so.1
genstub nvml libnvidia-ml.so.1

# These libraries are from the CUDA SDK and redistributed in the NuGet packages.
genstub cublas64_91 libcublas.so.9.1
genstub cufft64_91 libcufft.so.9.1
genstub curand64_91 libcurand.so.9.1
genstub cusolver64_91 libcusolver.so.9.1
genstub cusparse64_91 libcusparse.so.9.1
genstub nvgraph64_91 libnvgraph.so.9.1
genstub nvrtc64_91 libnvrtc.so.9.1
genstub nppc64_91 libnppc.so.9.1
genstub nppial64_91 libnppial.so.9.1
genstub nppicc64_91 libnppicc.so.9.1
genstub nppicom64_91 libnppicom.so.9.1
genstub nppidei64_91 libnppidei.so.9.1
genstub nppif64_91 libnppif.so.9.1
genstub nppig64_91 libnppig.so.9.1
genstub nppim64_91 libnppim.so.9.1
genstub nppist64_91 libnppist.so.9.1
genstub nppisu64_91 libnppisu.so.9.1
genstub nppitc64_91 libnppitc.so.9.1
genstub npps64_91 libnpps.so.9.1

# cuDNN redistribution is prohibited, but stub is generated to map DLL name.
genstub cudnn64_7 libcudnn.so.7.0

