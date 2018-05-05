#!/bin/bash

set -e 
cd "$(dirname "$0")"

genstub () 
{
    DLLNAME=$1
    TARGETS=$2
    LOCAL=$3

    if [ "$LOCAL" = "true" ]; then
        RPATH='-Wl,-rpath=$ORIGIN'
    else
        RPATH=''
    fi

    rm -f empty.c
    touch empty.c
    LIBARG=""
    for TARGET in $TARGETS ; do
        gcc -shared -o ${TARGET} empty.c
        LIBARG="$LIBARG -l:${TARGET}"
    done
    gcc -Wl,--no-as-needed $RPATH -shared -o lib${DLLNAME}.so -fPIC -L. $LIBARG
    for TARGET in $TARGETS ; do
        rm -f ${TARGET}
    done
    rm -f empty.c

    echo "Mapped ${DLLNAME}.dll ==> ${TARGETS}"
}

# nvcuda.dll or libcuda.so.1 and nvml.dll or libnvidia-ml.so.1 are always installed by the GPU driver.
genstub nvcuda libcuda.so.1 false
genstub nvml libnvidia-ml.so.1 false

# These libraries are from the CUDA SDK and redistributed in the NuGet packages.
genstub cublas64_91 libcublas.so.9.1 true
genstub cufft64_91 libcufft.so.9.1 true
genstub curand64_91 libcurand.so.9.1 true
genstub cusolver64_91 libcusolver.so.9.1 true
genstub cusparse64_91 libcusparse.so.9.1 true
genstub cudnn64_7 libcudnn.so.7.0 true
genstub nvgraph64_91 libnvgraph.so.9.1 true
genstub nvrtc64_91 "libnvrtc.so.9.1 libnvrtc-builtins.so" true
genstub nppc64_91 libnppc.so.9.1 true
genstub nppial64_91 libnppial.so.9.1 true
genstub nppicc64_91 libnppicc.so.9.1 true
genstub nppicom64_91 libnppicom.so.9.1 true
genstub nppidei64_91 libnppidei.so.9.1 true
genstub nppif64_91 libnppif.so.9.1 true
genstub nppig64_91 libnppig.so.9.1 true
genstub nppim64_91 libnppim.so.9.1 true
genstub nppist64_91 libnppist.so.9.1 true
genstub nppisu64_91 libnppisu.so.9.1 true
genstub nppitc64_91 libnppitc.so.9.1 true
genstub npps64_91 libnpps.so.9.1 true

