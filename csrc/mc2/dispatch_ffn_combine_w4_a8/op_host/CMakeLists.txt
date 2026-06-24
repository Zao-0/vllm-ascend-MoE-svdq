set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
endif()
option(SVDQ_W4A8_DEBUG_READBACK
       "Compile official W4A8 epilogues with FP32 GMM readback for isolated SVDQ validation."
       OFF)
set(_DISPATCH_FFN_W4A8_DEBUG_OPTS)
if(SVDQ_W4A8_DEBUG_READBACK)
    list(APPEND _DISPATCH_FFN_W4A8_DEBUG_OPTS -DW4A8_DEBUG)
endif()

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        dispatch_ffn_combine_w4_a8_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME DispatchFFNCombineW4A8
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -DHCCL_COMM
        -DCATLASS_ARCH=2201
        ${_DISPATCH_FFN_INC_OPTS}
        ${_DISPATCH_FFN_W4A8_DEBUG_OPTS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dispatch_ffn_combine_w4_a8 ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
