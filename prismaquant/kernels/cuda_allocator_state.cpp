#include <c10/core/AllocatorConfig.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("sizing", []() {
    using Config = c10::CachingAllocator::AcceleratorAllocatorConfig;
    return pybind11::make_tuple(Config::large_segment_size(),
                                Config::max_non_split_rounding_size());
  }, "Read effective allocator sizes through public C10 getters");
}
