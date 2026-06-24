set(_DISPATCH_FFN_SVDQ_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_SVDQ_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_SVDQ_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_SVDQ_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_SVDQ_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
endif()
list(APPEND _DISPATCH_FFN_SVDQ_INC_OPTS -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel)
option(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK
       "Mirror SVDQ low-rank MMAD FP32 L0C accumulators to GM for debug readback."
       OFF)
set(_DISPATCH_FFN_SVDQ_DEBUG_OPTS)
if(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)
    list(APPEND _DISPATCH_FFN_SVDQ_DEBUG_OPTS -DSVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)
endif()

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        dispatch_ffn_combine_w4_a8_svdq_def.cpp
        svdq_low_rank_debug_readback_def.cpp
        svdq_mixed_epilogue_debug_readback_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME DispatchFFNCombineW4A8SVDQ
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -DHCCL_COMM
        -DCATLASS_ARCH=2201
        ${_DISPATCH_FFN_SVDQ_INC_OPTS}
        ${_DISPATCH_FFN_SVDQ_DEBUG_OPTS}
)

add_ops_compile_options(
    OP_NAME SVDQLowRankDebugReadback
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -DCATLASS_ARCH=2201
        ${_DISPATCH_FFN_SVDQ_INC_OPTS}
        ${_DISPATCH_FFN_SVDQ_DEBUG_OPTS}
)

add_ops_compile_options(
    OP_NAME SVDQMixedEpilogueDebugReadback
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -DCATLASS_ARCH=2201
        ${_DISPATCH_FFN_SVDQ_INC_OPTS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE dispatch_ffn_combine_w4_a8_svdq svdq_low_rank_debug_readback svdq_mixed_epilogue_debug_readback
        ACLNNTYPE aclnn_inner aclnn_inner aclnn_inner
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
