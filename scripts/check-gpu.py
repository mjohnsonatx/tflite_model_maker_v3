#!/usr/bin/env python3
"""Check CUDA library loading and, when available, real GPU training operations."""
import argparse
import ctypes
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    arguments = parser.parse_args()
    if os.name == "nt":
        libraries = [
            "cudart64_110.dll", "cublas64_11.dll", "cublasLt64_11.dll",
            "cufft64_10.dll", "curand64_10.dll", "cusolver64_11.dll",
            "cusparse64_11.dll", "cudnn64_8.dll",
        ]
    else:
        libraries = [
            "libcudart.so.11.0", "libcublas.so.11", "libcublasLt.so.11",
            "libcufft.so.10", "libcurand.so.10", "libcusolver.so.11",
            "libcusparse.so.11", "libcudnn.so.8",
        ]
    failures = {}
    for library in libraries:
        try:
            ctypes.CDLL(library)
        except OSError as error:
            failures[library] = str(error)

    import numpy as np
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    report = {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "tensorflow_cuda_build": tf.sysconfig.get_build_info().get("cuda_version"),
        "library_errors": failures,
        "gpu_devices": [device.name for device in devices],
        "gpu_operations": "not run",
    }
    if not failures:
        cudnn = ctypes.CDLL(libraries[-1])
        cudnn.cudnnGetVersion.restype = ctypes.c_size_t
        report["cudnn_version"] = cudnn.cudnnGetVersion()

    if devices:
        runtime = ctypes.CDLL(libraries[0])
        runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
        runtime_version = ctypes.c_int()
        result = runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
        if result:
            raise RuntimeError("cudaRuntimeGetVersion failed with code " + str(result))
        report["cuda_runtime_version"] = runtime_version.value
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_soft_device_placement(False)
        with tf.device("/GPU:0"):
            product = tf.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
            image = tf.ones((1, 16, 16, 3))
            kernel = tf.Variable(tf.ones((3, 3, 3, 8)))
            with tf.GradientTape() as tape:
                convolution = tf.nn.conv2d(image, kernel, strides=1, padding="VALID")
                loss = tf.reduce_sum(convolution)
            gradient = tape.gradient(loss, kernel)
        np.testing.assert_allclose(product.numpy(), 64.0)
        np.testing.assert_allclose(loss.numpy(), 42336.0)
        np.testing.assert_allclose(gradient.numpy(), 196.0)
        for tensor in (product, convolution, gradient):
            if "GPU:0" not in tensor.device:
                raise RuntimeError("GPU operation ran on " + tensor.device)
        report["gpu_operations"] = "matrix multiply, convolution, and backpropagation passed"
    print(json.dumps(report, indent=2))
    if failures or (arguments.require_gpu and not devices):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
