#include <torch/extension.h>

#include <vector>

// 占位：后续接入 depthwise / fused IRB CUDA kernel
torch::Tensor depthwise_conv3x3_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 4, "input must be NCHW");

  // 暂时回退到 PyTorch 原生实现，保证接口可用
  const int64_t groups = weight.size(0);
  return torch::conv2d(input, weight, bias, /*stride=*/{stride, stride},
                       /*padding=*/{padding, padding}, /*dilation=*/{1, 1}, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depthwise_conv3x3_forward", &depthwise_conv3x3_forward,
        "Depthwise 3x3 conv forward (CUDA, stub)");
}
