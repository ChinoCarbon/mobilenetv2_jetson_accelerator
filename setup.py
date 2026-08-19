from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mnv2_cuda",
    ext_modules=[
        CUDAExtension(
            name="mnv2_cuda",
            sources=[
                "bindings/ops.cpp",
                "cuda/depthwise_conv.cu",
                "cuda/fused_irb.cu",
            ],
            include_dirs=["cuda/include"],
            extra_compile_args={
                "cxx": ["-O3"],
                # sm_89 = RTX 4090; sm_87 = Jetson Orin; 按实际硬件调整
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_89,code=sm_89",
                    "-gencode", "arch=compute_87,code=sm_87",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
