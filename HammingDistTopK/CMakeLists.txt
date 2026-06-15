# HammingDistTopK 自定义算子构建片段（CANN 8.5.0 ascendc 自定义算子工程）
#
# 用法：将本算子目录放入你的 custom op 工程（如 ops-math / CANN 提供的
# CustomOp 样例工程）的 op_host、op_kernel 对应位置，由工程顶层 CMake 统一
# 通过 ascendc_library / opbuild 收集编译。本文件给出独立编译时的参考。

cmake_minimum_required(VERSION 3.16)

# op_host：由 opbuild 工具编译为 *_op_proto / tiling 动态库
file(GLOB HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp)

# op_kernel：由 ascendc 编译为 device 侧二进制（按 soc_version=ascend910b）
file(GLOB KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/*.cpp)

# 实际集成时通常调用 CANN 提供的宏，例如：
#   add_ops_compile_options(HammingDistTopK ...)
#   add_ops_impl_target(...)
# 这里仅占位，提示需要把上述源文件挂到工程的算子编译目标上。
message(STATUS "HammingDistTopK host srcs:   ${HOST_SRC}")
message(STATUS "HammingDistTopK kernel srcs: ${KERNEL_SRC}")
