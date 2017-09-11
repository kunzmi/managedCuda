#!/bin/bash

set -e 
cd "$(dirname "$0")"

genstub () 
{
    rm -f empty.c
    touch empty.c
    DLLNAME=$1
    TARGET=$2
    gcc -shared -o ${TARGET} empty.c    
    gcc -Wl,--no-as-needed -shared -o lib${DLLNAME}.so -fPIC -L. -l:${TARGET}
    rm -f ${TARGET}
    rm -f empty.c

    echo "Mapped ${DLLNAME}.dll ==> ${TARGET}"
}


genstub nvcuda libcuda.so.1

genstub cublas64_80 libcublas.so.8.0
genstub cudnn64_6 libcudnn.so.6.0
genstub cufft64_80 libcufft.so.8.0
genstub curand64_80 libcurand.so.8.0
genstub cusolver64_80 libcusolver.so.8.0
genstub cusparse64_80 libcusparse.so.8.0
genstub nppi64_80 libnppi.so.8.0
genstub nppc64_80 libnppc.so.8.0
genstub npps64_80 libnpps.so.8.0
genstub nvgraph64_80 libnvgraph.so.8.0
genstub nvrtc64_80 libnvrtc.so.8.0

