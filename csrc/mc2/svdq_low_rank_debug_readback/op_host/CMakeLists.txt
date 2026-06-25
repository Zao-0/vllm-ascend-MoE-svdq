add_op_to_compiled_list()

set(_SVDQ_DISPATCH_OP dispatch_ffn_combine_w4_a8_svdq)
set(_SVDQ_DISPATCH_SELECTED OFF)
if(NOT ASCEND_OP_NAME OR "${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
    set(_SVDQ_DISPATCH_SELECTED ON)
else()
    list(FIND ASCEND_OP_NAME ${_SVDQ_DISPATCH_OP} _SVDQ_DISPATCH_INDEX)
    if(NOT _SVDQ_DISPATCH_INDEX EQUAL -1)
        set(_SVDQ_DISPATCH_SELECTED ON)
    endif()
endif()

if(NOT _SVDQ_DISPATCH_SELECTED)
    set(_SVDQ_DISPATCH_HOST_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/../../${_SVDQ_DISPATCH_OP}/op_host)
    set(_SVDQ_DISPATCH_KERNEL_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/../../${_SVDQ_DISPATCH_OP}/op_kernel)

    set(_SVDQ_DEBUG_INC_OPTS)
    if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
        list(APPEND _SVDQ_DEBUG_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
        list(APPEND _SVDQ_DEBUG_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
        list(APPEND _SVDQ_DEBUG_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    endif()
    if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
        list(APPEND _SVDQ_DEBUG_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    endif()
    list(APPEND _SVDQ_DEBUG_INC_OPTS -I${_SVDQ_DISPATCH_KERNEL_DIR})

    option(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK
           "Mirror SVDQ low-rank MMAD FP32 L0C accumulators to GM for debug readback."
           ON)
    set(_SVDQ_DEBUG_COMPILE_OPTS)
    if(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)
        list(APPEND _SVDQ_DEBUG_COMPILE_OPTS -DSVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)
    endif()

    add_ops_compile_options(
        OP_NAME SVDQLowRankDebugReadback
        OPTIONS
            --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            -DCATLASS_ARCH=2201
            ${_SVDQ_DEBUG_INC_OPTS}
            ${_SVDQ_DEBUG_COMPILE_OPTS}
    )

    add_modules_sources(
        OPTYPE svdq_low_rank_debug_readback
        ACLNNTYPE aclnn_inner
    )

    if (BUILD_OPEN_PROJECT)
        target_sources(op_host_aclnnInner PRIVATE
            ${_SVDQ_DISPATCH_HOST_DIR}/svdq_low_rank_debug_readback_def.cpp
        )
    endif()
    target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
        ${_SVDQ_DISPATCH_HOST_DIR}/svdq_low_rank_debug_readback_def.cpp
    )
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        ${_SVDQ_DISPATCH_HOST_DIR}/op_api/aclnn_svdq_lowrank_debug_readback.cpp
    )
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE
        ${_SVDQ_DISPATCH_HOST_DIR}/op_api/aclnn_svdq_lowrank_debug_readback.h
    )

    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${_SVDQ_DISPATCH_HOST_DIR}/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${_SVDQ_DISPATCH_HOST_DIR}
        ${_SVDQ_DISPATCH_KERNEL_DIR}
    )
endif()
