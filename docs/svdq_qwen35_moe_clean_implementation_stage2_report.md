# SVDQ Qwen3.5 MoE Clean Implementation - Stage 2 Report

## Authoritative Status - 2026-06-26

This table supersedes ambiguous status statements in older handoff sections. Older sections are historical evidence unless explicitly referenced by the latest active section.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Built, installed, registered, and launched on logical NPU 0 in prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden exact-match gate passed with mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Rebuilt loop-stats diagnostic proves the official GMM2-only entry path executes, but `cumsumMM` is zero at scheduling time, so official GMM2 schedules zero active tiles. Gate B/Gate C remain blocked. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B/Gate C. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | Host tiling must remain fail-closed until Stage 2.2+ production gates pass. |

## Current State - 2026-06-26T12:28Z

Status: Stage 2.2 remains FAIL / IN PROGRESS.

Binding prompt and appendix state:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` remains the active working prompt.
- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was read and is binding for this handoff.
- The public `torch_npu.npu_grouped_matmul` path was not modified, debugged, or used as progress.
- The official `dispatch_ffn_combine_w4_a8` producer/dequant path remains the only source of truth for W4A8 GMM2.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

New debug change:

- Added a debug-only GMM2 loop-stats sentinel to `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`.
- The sentinel is enabled by `430000.0f < swigluLimit < 440000.0f`; the current probe uses `--swiglu-limit 434343`.
- In GMM2-only mode it writes scheduling state to `ptrDebugGMM2` and returns before MMAD/dequant. It does not change the production SVDQ operator.
- The probe parser in `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py` records `loop_stats` and marks the diagnostic as non-passing numerical evidence by design.

Build and install evidence:

- First loop-stats probe before forcing the generated kernel rebuild was invalid as code-behavior evidence: the copied generated kernel source did not yet contain the loop-stats marker.
- Forced kernel rebuild initially exposed an AscendC compile restriction: unsigned-counter to `float` casts in AIC/AIV code are rejected.
- Fixed the diagnostic by casting unsigned counters through `int32_t` before writing the float debug buffer.
- Forced target rebuild:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_build_kernel_after_cast_fix.log`
  - Log scan found no `error:`, `[ERROR]`, OPC failure, `ld.lld: error`, or old unsigned-cast failure markers.
  - Generated `.o`/`.json` files were refreshed from `2026-06-26 12:17:05` through `2026-06-26 12:22:19` UTC.
- Staged install:
  - `ascendc_ops_config.py`: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_after_cast_fix_ops_config.log`
  - `cmake --install`: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_after_cast_fix_cmake_install.log`
  - Correct install invocation must run from the staged root. Two earlier wrong-path/wrong-cwd install attempts are retained as install-script path evidence, not device evidence.
  - Repo-local install from staged root: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_after_cast_fix_install_repo_root_cwd.log`, result `SUCCESS`.
  - System OPP install from staged root: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_after_cast_fix_install_system_opp_root_cwd.log`, result `SUCCESS`.

Validation evidence:

- Device boundary:
  - `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 npu-smi info`
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_after_cast_fix_npu_smi.log`
  - Physical NPUs `0,1,2,3` are visible, all `910B4`, health `OK`, no running NPU processes.
- Torch/NPU visibility:
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_after_cast_fix_torch_npu_visibility.log`
  - `npu_available: True`, `visible_device_count: 4`, logical devices `0..3` are `Ascend910B4`.
- ABI registration:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_after_cast_fix_abi.log`
  - Result: `1 passed, 16 warnings`.
- Loop-stats probe:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ... python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --swiglu-limit 434343 --summary-name phase_stage2_gmm2_loop_stats_probe_after_cast_fix.json`
  - Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_probe_after_cast_fix.log`
  - Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_loop_stats_probe_after_cast_fix.json`
  - Result: diagnostic `passed: false` by design; `loop_stats.magic == 434343.0`, proving the official GMM2-only entry path executed.
  - Gate A remains valid: `hidden_packed_exact: true`, `hidden_packed_mismatch_count_zero: true`, `hidden_scale_finite: true`, `hidden_scale_nonzero: true`, `canonical_hidden_finite: true`, `canonical_hidden_nonzero: true`.
  - New blocker: `total_active_rows: 0`, `total_core_loops: 0`, `groups_with_work: 0`; every group reports `raw_current_m: 0`, `clipped_current_m: 0`, and `core_loops: 0`.

Interpretation:

- The loop-stats diagnostic converts the previous all-zero GMM2 output from an ambiguous MMAD/dequant/readback failure into a token-state/scheduling boundary failure.
- Official GMM2 entry is reached, but the state visible through `cumsumMM` at scheduling time contains zero rows for every local expert.
- Because no active tiles are scheduled, a raw-C2 follow-up would not be meaningful until the official-path GMM2-only debug op seeds or reuses the same token-count/cumsum lifecycle as the successful official W4A8 path.
- Stage 2.2 Gate B and Gate C remain failed.

## Baseline - 2026-06-26T06:17Z

Status: IN PROGRESS.

Historical note: this baseline section is superseded by the latest `Authoritative Status` table and later handoff sections. It is retained as evidence history.

Working prompt:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` is the active Stage 2 prompt.
- The prompt names `svdq_qwen35_moe_clean_implementation_report_clarified_completed.md` as the authoritative baseline, but that file is not present under `/root/workspace/lza` in this environment. Until it is supplied, the working baseline is the current pushed repo commit plus the Stage 2 prompt, `svdq_qwen35_moe_clean_implementation_prompt.md`, Appendix 5, and the older long report for historical details only.

Repository baseline:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- HEAD: `e3b012efa03ab9dca7ce44c12f0387d8443b5091`
- Describe: `v0.20.2rc1-123-ge3b012ef`
- Last commit: `e3b012ef Add packed hidden INT4 mixed epilogue debug ABI`
- Installed Python package path: `/root/workspace/lza/vllm-ascend/vllm_ascend/__init__.py`
- Submodule baseline: `csrc/third_party/catlass` at `41bf90da655bba3c66d0acd7e00abe33960ecfd6`

Pre-existing dirty state left untouched:

- `csrc/utils/inc/kernel/moe_distribute_base.h`
- `csrc/build_out/`
- `extra-info/`

Visible NPU boundary:

- Command used: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 npu-smi info`
- Visible physical NPUs: `0,1,2,3`, all `910B4`, health `OK`.
- `torch_npu` check with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`:
  - `npu_available: True`
  - `visible_device_count: 4`
  - logical devices: `0 Ascend910B4`, `1 Ascend910B4`, `2 Ascend910B4`, `3 Ascend910B4`
- No devices `4-7` were exposed to the process.

## Stage 2.0 - Generated Seven-Output Mixed Epilogue Debug Op

Status: PARTIAL PASS.

Required ABI order:

1. `gate_up_total`
2. `hidden_bf16`
3. `hidden_int8`
4. `hidden_int4_packed`
5. `hidden_scale`
6. `down_total`
7. `out_bf16`

Official hidden INT4 source paths inspected:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - `FetchAndPreprocessInt8ToInt4`
  - Rereads the hidden INT8 plus scale row payload, copies scale to GMM2 scale GM, packs high INT4 to `gmA1I4_I8[absStartAddr]`, and packs low INT4 to `gmA1I4_I8[absStartAddr + hiddenSize / 2]`.
  - High path: cast INT8 to half, multiply by `0.0625`, floor-cast to `int4b_t`, copy `hiddenSize / 2` bytes.
  - Low path: bitwise `AND` INT8 reinterpret with `0x0F0F`, cast through half, add `-8`, cast to `int4b_t`, copy `hiddenSize / 2` bytes.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp`
  - Official first epilogue quantizes the FP32 SwiGLU hidden, writes the per-token scale, writes high INT4 to the first half of the row, and low INT4 to the second half.

SVDQ-specific adaptation under validation:

- The mixed debug op writes canonical SVDQ-modified `hidden_bf16` to GM.
- It writes the plain debug `hidden_int8` for inspection.
- It additionally writes `hidden_int4_packed` using the official high/low packed hidden boundary layout so the next Stage 2 gate can verify exact device bytes before any modified-hidden W4A8 GMM2 launch.

Focused build evidence:

- `cmake --build csrc/build --target generate_compile_cmd_ascend910b -- -B -j1`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_generate_compile_cmd_ascend910b.log`
  - This regenerated `csrc/build/binary/ascend910b/gen/SVDQMixedEpilogueDebugReadback_8f2440b95006b27937d5cc3437e43193_param.json`.
  - The regenerated param JSON has seven outputs in the required order: `gateUpTotal`, `hiddenBf16`, `hiddenInt8`, `hiddenInt4Packed`, `hiddenScale`, `downTotal`, `outBf16`.
- `cmake --build csrc/build --target svdq_mixed_epilogue_debug_readback_ascend910b_0 -- -B -j1`
  - Result: pass after compile-command regeneration.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_build_svdq_mixed_epilogue_debug_readback_ascend910b_0_retry.log`
  - Produced:
    - `csrc/build/binary/ascend910b/bin/svdq_mixed_epilogue_debug_readback/SVDQMixedEpilogueDebugReadback_8f2440b95006b27937d5cc3437e43193.json`
    - `csrc/build/binary/ascend910b/bin/svdq_mixed_epilogue_debug_readback/SVDQMixedEpilogueDebugReadback_8f2440b95006b27937d5cc3437e43193.o`

Root cause of the earlier Stage 2.0 compile failure:

- The first retry built the single binary target while `csrc/build/binary/ascend910b/gen/SVDQMixedEpilogueDebugReadback-svdq_mixed_epilogue_debug_readback-0.sh` still referenced stale generated metadata.
- The stale shell expected `SVDQMixedEpilogueDebugReadback_8e345e1e8f3b5ad4f76c5d68573e5467_param.json`, which did not exist.
- Running the supported `generate_compile_cmd_ascend910b` target regenerated the compile shell and param JSON with the current seven-output ABI hash `8f2440b95006b27937d5cc3437e43193`.
- No generated `*_param.json` was fabricated or hand-patched.

Install and launch status:

- Not complete.
- The source-tree package mirror under `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/...` is still stale and untracked by git except for `.gitkeep`.
- Current stale installed dynamic file still advertises six outputs and is missing `hiddenInt4Packed`:
  - `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/svdq_mixed_epilogue_debug_readback.py`
- A broad `ops_transformer_config` refresh was started to regenerate package binary config, but it began compiling the full selected operator set and was intentionally interrupted before completion:
  - Command: `cmake --build csrc/build --target ops_transformer_config -- -B -j1`
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_ops_transformer_config.log`
  - Result: interrupted with `SIGINT`, not a mixed-epilogue compile failure.
- Because install/register was not completed, no real-device launch or packed-hidden probe was run after this compile pass.

Current next action after environment rebuild:

- Rebuild through the supported flow in this order:
  1. `cmake --build csrc/build --target generate_compile_cmd_ascend910b -- -B -j1`
  2. `cmake --build csrc/build --target svdq_mixed_epilogue_debug_readback_ascend910b_0 -- -B -j1`
  3. Complete the supported package/config install path so the package mirror also has the seven-output dynamic Python and seven-output binary config.
- Confirm installed metadata are seven-output before launching:
  - Dynamic Python output list includes `hiddenInt4Packed`.
  - Binary config simplified/static keys encode seven outputs.
  - Torch schema/meta returns seven tensors.
- Launch `svdq_mixed_epilogue_debug_readback` on logical NPU 0.
- Only then run:
  - `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_mixed_epilogue_device_probe.py --require-npu --summary-name phase_mixed_epilogue_packed_i4_summary.json`
- The Stage 2.1 packed-hidden exact-match gate remains pending until that probe reports exact equality and zero mismatches.

## Current Handoff State - 2026-06-26T06:52Z

Production state:

- `DispatchFFNCombineW4A8SVDQ` remains fail-closed.
- No public `torch_npu.npu_grouped_matmul` path was modified, debugged, or used as progress for this gate.
- No scale guessing, repacking workaround, scalar substitute, or public grouped-matmul experiment was treated as progress.

Committed source state at handoff:

- Last pushed implementation commit before this report update: `e3b012efa03ab9dca7ce44c12f0387d8443b5091`.
- The source implements the seven-output debug ABI and the packed hidden INT4 debug output path.
- Focused host tests from the previous commit passed:
  - `tests/ut/quantization/test_svdq_post_load.py::test_svdq_mixed_epilogue_reference_adds_residual_and_lowrank_before_swiglu`
  - `tests/ut/ops/test_svdq_moe_abi.py::test_svdq_mixed_epilogue_debug_torch_schema_meta_and_adapter_are_registered`
  - `tests/ut/ops/test_svdq_moe_abi.py::test_svdq_mixed_epilogue_device_probe_matches_reference_oracle`

Dirty state intentionally left untouched:

- `csrc/utils/inc/kernel/moe_distribute_base.h`
- `csrc/build_out/`
- `extra-info/`

Blocking boundary:

- Stage 2.0 source and focused binary compile are now validated.
- Stage 2.0 install/register/launch is still incomplete.
- Stage 2.1 packed-hidden exact-match validation has not started in this environment.

## Current Handoff State - 2026-06-26T07:16Z

Status: IN PROGRESS.

Working prompt:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` was read and is the active working prompt.
- The authoritative unresolved boundary remains:
  `SVDQ-modified canonical BF16 hidden -> official hidden quantization/high-low packed INT4 -> official W4A8 GMM2`.
- The BF16 low-rank producer path was not reopened.
- No public `torch_npu.npu_grouped_matmul` path was modified, debugged, or used as progress.

Repository state:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- HEAD before this report update: `2f9cdace6891008302c5e8cbfb412f8f97e29e78`
- Origin: `git@github.com:Zao-0/vllm-ascend-MoE-svdq.git`
- Pre-existing dirty/untracked paths left untouched:
  - `csrc/utils/inc/kernel/moe_distribute_base.h`
  - `csrc/build_out/`
  - `extra-info/`

Stage 2.0 package/config install evidence:

- Direct project metadata generator was run after the focused binary build:
  - Command: `/usr/local/python3.12.13/bin/python3.12 csrc/cmake/scripts/util/ascendc_ops_config.py -p csrc/build/binary/ascend910b/bin -s ascend910b`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_direct_ascendc_ops_config.log`
  - Generated build-tree config files:
    - `csrc/build/binary/ascend910b/bin/binary_info_config.json`
    - `csrc/build/binary/ascend910b/bin/relocatable_kernel_info_config.json`
    - `csrc/build/binary/ascend910b/bin/svdq_mixed_epilogue_debug_readback.json`
  - The generated config contains `hiddenInt4Packed` and hash `8f2440b95006b27937d5cc3437e43193`.
- CMake install probe:
  - Command: `cmake --install csrc/build --prefix /tmp/svdq_install_probe2`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_cmake_install_probe2.log`
  - Installed package probe contained the seven-output dynamic Python file, kernel config, kernel JSON, and kernel object.
- Repository custom-op mirror install:
  - Command: `(cd /tmp/svdq_install_probe2 && ./install.sh --install-path=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom)`
  - Result: pass, installer printed `SUCCESS`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_install_probe2_to_repo_custom_ops.log`
  - Installed dynamic file now advertises seven outputs:
    `gateUpTotal`, `hiddenBf16`, `hiddenInt8`, `hiddenInt4Packed`, `hiddenScale`, `downTotal`, `outBf16`.
  - Installed config files contain `hiddenInt4Packed` and hash `8f2440b95006b27937d5cc3437e43193`:
    - `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b/svdq_mixed_epilogue_debug_readback.json`
    - `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json`
  - Installed kernel artifacts exist:
    - `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/svdq_mixed_epilogue_debug_readback/SVDQMixedEpilogueDebugReadback_8f2440b95006b27937d5cc3437e43193.json`
    - `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/svdq_mixed_epilogue_debug_readback/SVDQMixedEpilogueDebugReadback_8f2440b95006b27937d5cc3437e43193.o`
  - `libcust_opapi.so` exports:
    - `aclnnSVDQMixedEpilogueDebugReadback`
    - `aclnnSVDQMixedEpilogueDebugReadbackGetWorkspaceSize`
    - `aclnnInnerSVDQMixedEpilogueDebugReadback`
    - `aclnnInnerSVDQMixedEpilogueDebugReadbackGetWorkspaceSize`

Registration and launch evidence:

- Focused source/adapter pytest:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_mixed_epilogue_debug_torch_schema_meta_and_adapter_are_registered`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_pytest_mixed_epilogue_schema_meta_adapter.log`
- First real-device launch attempt:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_mixed_epilogue_device_probe.py --require-npu --summary-name phase_mixed_epilogue_packed_i4_summary.json`
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_mixed_epilogue_device_probe_packed_i4.log`
  - Result: fail before kernel execution.
  - The call reached `aclnnSVDQMixedEpilogueDebugReadback`, then ACL returned:
    `AclNN_Inner_Error(EZ9999): The binary bin not found`.
  - The failure occurred during `NnopbaseExecutorTilingAndUpdateBinInfo` / `NnopbaseExecutorMatchCache` / `NnopbaseRunForWorkspace`.
- Second real-device launch attempt with custom OPP variables set before Python startup:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_mixed_epilogue_device_probe.py --require-npu --summary-name phase_mixed_epilogue_packed_i4_summary_envpreset.json`
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_mixed_epilogue_device_probe_packed_i4_envpreset.log`
  - Result: same failure, `The binary bin not found`.

Current interpretation:

- Stage 2.0 source build, package generation, repository custom-op install, op API symbol availability, and focused schema/meta/adapter registration are validated.
- Stage 2.0 is not complete because the real logical-NPU-0 launch still fails before kernel execution.
- The remaining blocker is custom-op binary/config discovery or runtime descriptor-to-binary matching for `SVDQMixedEpilogueDebugReadback`, not W4A8 math, hidden packing correctness, or BF16 low-rank AIC behavior.
- Stage 2.1 packed-hidden exact-match validation remains pending; no packed-hidden numerical evidence was produced in this environment because the launch did not reach kernel execution.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

Recommended next action after environment rebuild:

1. Recreate the focused generated build state for `ascend910b` using the supported generation/build flow.
2. Reinstall the package through the supported installer and confirm the seven-output installed metadata.
3. Before launching, verify which custom OPP vendor path ACL actually resolves when both the system vendor path and repository-local `ASCEND_CUSTOM_OPP_PATH` exist.
4. Continue debugging the `The binary bin not found` launch failure from ACL custom-op binary discovery and descriptor matching.
5. Do not proceed to Stage 2.1 or GMM2 integration until `svdq_mixed_epilogue_debug_readback` launches on logical NPU 0.

## Current Handoff State - 2026-06-26T07:45Z

Status: IN PROGRESS.

Current resolved boundary:

- The Stage 2.0 custom-op discovery failure is resolved in this environment.
- The seven-output ABI is now consistent across generated kernel metadata, installed dynamic files, `libcust_opapi.so`, and the in-place Python extension.
- The real logical-NPU-0 mixed epilogue probe launches and passes, including exact packed hidden INT4 readback.

Root cause fixed during this handoff:

- The first launch blocker, `AclNN_Inner_Error(EZ9999): The binary bin not found`, persisted after installing the seven-output package because `libcust_opapi.so` was still a stale six-output wrapper.
- `strings csrc/build/libcust_opapi.so` originally showed no `hiddenInt4PackedOut` and the old output order.
- Rebuilding the supported generated host/opapi targets fixed this layer:
  - `cmake --build csrc/build --target ops_aclnn -- -B -j1`
  - `cmake --build csrc/build --target cust_opapi -- -B -j1`
  - Evidence:
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_rebuild_ops_aclnn_for_mixed7.log`
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_rebuild_cust_opapi_for_mixed7.log`
- A refreshed install package was generated and installed into both the repo-local custom OPP mirror and the system OPP:
  - `cmake --install csrc/build --prefix /tmp/svdq_install_probe3`
  - `(cd /tmp/svdq_install_probe3 && ./install.sh --install-path=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom)`
  - `(cd /tmp/svdq_install_probe3 && ./install.sh --install-path=/usr/local/Ascend/cann-9.0.0/opp)`
  - Evidence:
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_cmake_install_probe3_after_opapi_rebuild.log`
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_install_probe3_to_repo_custom_ops.log`
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_install_probe3_to_system_opp.log`
- The refreshed `libcust_opapi.so` checksum in build, repo-local install, and system install is:
  - `12b00f273661368fecca86e52736f257668f23b4f4ef1cc41fd4a35e4957dd69`
- The second launch blocker was a stale in-place Python extension. It still advertised the old six-output torch schema and then segfaulted in `NnopbaseMatchArgs` after the opapi layer was updated.
- Rebuilding and installing the extension from the existing CMake build tree fixed this layer:
  - `cmake --build build/temp.linux-aarch64-cpython-312 --target vllm_ascend_C -- -B -j1`
  - `cmake --install build/temp.linux-aarch64-cpython-312`
  - Evidence:
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_rebuild_vllm_ascend_C_mixed7.log`
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_install_vllm_ascend_C_mixed7.log`
- The rebuilt extension schema includes:
  - `hidden_int4_packed` between `hidden_int8` and `hidden_scale`.

Validation after the fixes:

- Focused schema/meta/adapter pytest:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_mixed_epilogue_debug_torch_schema_meta_and_adapter_are_registered`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_pytest_mixed_epilogue_schema_meta_adapter_after_extension_rebuild.log`
- Real-device mixed epilogue probe:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_mixed_epilogue_device_probe.py --require-npu --summary-name phase_mixed_epilogue_packed_i4_summary_after_extension_rebuild.json`
  - Result: pass.
  - Evidence log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_mixed_epilogue_device_probe_after_extension_rebuild.log`
  - Evidence summary: `/root/workspace/lza/svdq_clean_evidence/phase_mixed_epilogue_packed_i4_summary_after_extension_rebuild.json`

Numerical evidence from the passing probe:

- Environment:
  - `npu_device_count: 4`
  - `selected_device: 0`
  - `runtime_soc_version: Ascend910B4`
- Appendix 3 gate sequence:
  - `residual_only`: pass.
  - `svdq_only`: pass.
  - `two_branch_nonzero`: pass.
- Exact hidden quantization/readback checks:
  - `hidden_q_exact_match: true`
  - `hidden_q_mismatch_count: 0`
  - `hidden_q_max_abs_diff: 0`
  - `hidden_q_packed_exact_match: true`
  - `hidden_q_packed_mismatch_count: 0`
  - `hidden_q_packed_max_abs_diff: 0`
- Final probe result:
  - `passed: true`

Current remaining boundary:

- Stage 2.0 launch/register is now validated.
- Stage 2.1 packed hidden INT4 readback is validated for the mixed epilogue debug operator.
- The next unresolved gate is the official W4A8 GMM2 consumer boundary:
  `SVDQ-modified canonical BF16 hidden -> official hidden quantization/high-low packed INT4 -> official W4A8 GMM2`.
- The next work should reuse or isolate the official `dispatch_ffn_combine_w4_a8` GMM2 consumer path against the packed hidden buffer that has now been validated.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.
- No public `torch_npu.npu_grouped_matmul` path was modified, debugged, or used as progress.

Git/worktree state to preserve across rebuild:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- HEAD before this report update: `2ac7af0a47d5bae9687d4d55579791b44fe3de3a`
- Pre-existing dirty/untracked paths intentionally left untouched:
  - `csrc/utils/inc/kernel/moe_distribute_base.h`
  - `csrc/build_out/`
  - `extra-info/`
- Generated custom-op install mirrors under `vllm_ascend/_cann_ops_custom` were refreshed for local validation but are not intended to be committed except for tracked placeholders.

## Current Handoff State - 2026-06-26T08:40Z

Status: IN PROGRESS.

Current working prompt:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` was reread and remains the active working prompt.
- The immediate unresolved Stage 2.2 boundary is:
  `validated SVDQ-modified packed hidden INT4 + per-token hidden scale -> official W4A8 GMM2 consumer path`.
- The public `torch_npu.npu_grouped_matmul` path was not modified, reinterpreted, debugged, or used as progress.
- No scale-guessing, repacking workaround, scalar substitute, or public grouped-matmul probe is counted as progress.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

Repository state before this handoff commit:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- HEAD before this report update: `484a40f6f3b7ee78c19ffea6a553ffda6c0163b1`
- Origin: `git@github.com:Zao-0/vllm-ascend-MoE-svdq.git`
- Pre-existing dirty/untracked paths still intentionally left out of this work:
  - `csrc/utils/inc/kernel/moe_distribute_base.h`
  - `csrc/build_out/`
  - `extra-info/`

New isolated Stage 2.2 implementation work:

- Added a debug-only W4A8 GMM2 relaunch path that seeds the official W4A8 internal packed-hidden buffers from an externally validated packed INT4 hidden tensor and hidden scale tensor.
- The path enters the official `DispatchFFNCombineW4A8` kernel with `gmm2OnlyFromPacked` enabled.
- AIC copies the supplied packed hidden bytes into official `gmA2I4_I8`, copies the supplied per-token hidden scales into official `gmPerTokenScale2`, copies external expert-token counts into official `tokenPerExpert`, builds the official `cumsumMM`, and then calls the official `GMM2(params)`.
- AIV only emits the synchronization flags needed to unblock the official GMM2 wait points; it does not implement a second GMM2 or public grouped-matmul substitute.
- This intentionally skips redoing hidden quantization inside the new operator. The input packed hidden boundary is expected to come from the already validated Stage 2.1 mixed epilogue debug op.

Source files added:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/svdqw4_a8_gmm2_debug_readback.cpp`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/svdqw4_a8_gmm2_debug_readback_def.cpp`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_gmm2_debug_readback.h`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_gmm2_debug_readback.cpp`
- `csrc/mc2/svdq_w4a8_gmm2_debug_readback/svdq_w4a8_gmm2_debug_readback_torch_adpt.h`

Source files modified:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/CMakeLists.txt`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp`
- `csrc/build_aclnn.sh`
- `csrc/torch_binding.cpp`
- `csrc/torch_binding_meta.cpp`
- `tests/ut/ops/test_svdq_moe_abi.py`

New torch op surface:

- `torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback(...) -> Tensor`
- Inputs include the original official W4A8 argument set plus:
  - `hidden_x_int4_packed`
  - `hidden_x_scale`
  - `external_expert_token_nums`
- Output is the official debug GMM2 post-dequant readback tensor.

Build and static validation evidence:

- Generate compile commands:
  - Command: `cmake --build csrc/build --target generate_compile_cmd_ascend910b -- -B -j1`
  - First result: fail because the new opdef dtype/format option lists were inconsistent.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_build_gmm2_debug_generate_compile_cmd.log`
  - Fix: align all input/output dtype and format option-list lengths with the existing official W4A8 debug opdef.
- Generate compile commands retry:
  - Command: `cmake --build csrc/build --target generate_compile_cmd_ascend910b -- -B -j1`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_build_gmm2_debug_generate_compile_cmd_retry.log`
- Focused Ascend910B kernel build:
  - Command: `cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`
  - Result: pass across generated variants `_0` through `_7`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_build_svdqw4a8_gmm2_debug_readback_ascend910b.log`
- Focused torch schema/meta/adapter test:
  - Command: `pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`
  - Result: pass, `1 passed, 16 warnings`.

Current limitation:

- This handoff stops at source implementation, focused CANN kernel build, and torch registration/static validation because the environment is about to be rebuilt.
- The new GMM2 debug op has not yet been installed into the refreshed custom OPP package, rebuilt into the in-place Python extension, launched on logical NPU 0, or compared numerically against the unfused official-GMM2 reference.
- Therefore the Stage 2.2 real-device numerical gate is still pending.

Required next steps after environment rebuild:

1. Keep `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` and reconfirm exactly four visible logical NPUs.
2. Regenerate/build/install the new custom op through the supported flow:
   - `cmake --build csrc/build --target generate_compile_cmd_ascend910b -- -B -j1`
   - `cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`
   - rebuild `ops_aclnn`, `cust_opapi`, and the in-place `vllm_ascend_C` extension before launch.
3. Add or run a Stage 2.2 device probe that feeds the Stage 2.1 validated mixed hidden packed bytes and scales into `svdq_w4a8_gmm2_debug_readback`.
4. Compare the returned official GMM2 post-dequant tensor against the unfused official-GMM2 reference using real-checkpoint W2/scales and same routing.
5. Only after the real-device GMM2 gate passes should work proceed to down projection SVDQ composition, final combine, or production fused integration.

## Current Handoff State - 2026-06-26T09:55Z

Status: IN PROGRESS.

Working prompt and constraints:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` remains the active working prompt.
- The active Stage 2.2 gate is still:
  `validated SVDQ-modified packed hidden INT4 + per-token hidden scale -> official W4A8 GMM2 AIC/AIV consumer path`.
- The public `torch_npu.npu_grouped_matmul` path was not modified, reinterpreted, debugged, or used as progress.
- No scale-guessing, repacking workaround, scalar substitute, or public grouped-matmul probe is counted as progress.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.
- Visible-device boundary was preserved with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`; the real-device probe saw 4 visible NPUs and `Ascend910B4` on logical device 0.

New source state in this handoff:

- Added `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`.
- Modified the isolated W4A8 GMM2 debug path in `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`.
- The new probe feeds the already validated Stage 2.1 mixed-epilogue hidden boundary into `torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback`.
- The probe validates:
  - canonical BF16 mixed hidden is finite and nonzero;
  - plain hidden INT8 is finite and nonzero;
  - packed hidden INT4 bytes exactly match `pack_official_hidden_i4_reference`;
  - per-token hidden scale is finite and nonzero;
  - real checkpoint W2 and W2 scale tensors are loaded through the official ModelSlim postload path;
  - the returned official GMM2 post-dequant readback is compared with the unfused official-contract GMM2 reference.
- The debug kernel AIV branch was changed from signal-only to `GMM2OnlyDequantReadback`, which constructs the official `BlockEpilogue2`, calls `SignalGMM2OnlyReady`, and then enters official `CombineV2`.
- The GMM2-only setup now zeroes `preSumBeforeRank` before `CombineV2` so the single-rank debug path does not consume stale rank-prefix data.

Rebuild and install evidence after the AIV consumer change:

- Focused debug kernel build:
  - Command: `cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_kernel_build_aiv_consumer_fix_retry.log`
- `ops_aclnn` rebuild:
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_rebuild_ops_aclnn_aiv_consumer_fix.log`
- `cust_opapi` rebuild:
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_rebuild_cust_opapi_aiv_consumer_fix.log`
- Custom-op package install:
  - Repo-local custom OPP install: pass, `SUCCESS`.
  - System OPP install: pass, `SUCCESS`.
  - Evidence:
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_install_repo_custom_ops_aiv_consumer_fix.log`
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_install_system_opp_aiv_consumer_fix.log`
- In-place Python extension rebuild and install:
  - Build result: pass.
  - Install result: pass.
  - Evidence:
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_rebuild_vllm_ascend_C_aiv_consumer_fix.log`
    - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_install_vllm_ascend_C_aiv_consumer_fix.log`
- ABI registration test:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`
  - Result: pass, `1 passed, 16 warnings`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_pytest_aiv_consumer_fix.log`

Stage 2.2 real-device probe result:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --summary-name phase_stage2_gmm2_from_mixed_hidden_aiv_consumer_fix.json`
- Result: fail, `passed: false`.
- Evidence:
  - Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_from_mixed_hidden_probe_aiv_consumer_fix.log`
  - Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_from_mixed_hidden_aiv_consumer_fix.json`
- Validated healthy inputs:
  - `torch_op_registered: true`
  - `official_gmm2_kernel_launched: true`
  - `public_grouped_matmul_used: false`
  - `production_svdq_host_tiling_fail_closed: true`
  - hidden packed exact match: `true`
  - hidden packed mismatch count: `0`
  - canonical hidden BF16 finite/nonzero, `max_abs: 4.3125`
  - hidden INT8 finite/nonzero, `max_abs: 127`
  - hidden scale finite/nonzero, `max_abs: 0.03395669162273407`
- Failing boundary:
  - GMM2 post-dequant active tensor remains all zeros:
    - `nonzero: false`
    - `max_abs: 0.0`
    - `mean_abs: 0.0`
  - Unfused official-contract reference is nonzero.
  - Reference comparison over 64 rows:
    - `max_abs: 0.4712103307247162`
    - `mean_abs: 0.060490094125270844`
    - tolerances: `max_abs <= 0.0002`, `mean_abs <= 0.00002`

Current diagnosis:

- The Stage 2.1 packed-hidden boundary remains valid and should not be re-debugged as a scale or packing problem.
- The isolated Stage 2.2 debug op reaches registration, build, install, launch, and healthy input validation, but still does not surface nonzero official GMM2 post-dequant output.
- The current likely remaining issue is inside the isolated GMM2-only AIV/AIC producer-consumer wiring: `CombineV2`/`BlockEpilogue2` is now entered, but the readback buffer still observes zero data.
- Next work should inspect only the official-kernel-derived GMM2-only synchronization/state needed by `CombineV2` and `BlockEpilogue2`:
  - C2V flag wait/finalize ordering;
  - `cumsumMM`, `tokenPerExpert`, and `preSumBeforeRank` layout for the single-rank debug path;
  - whether the GMM2-only path is writing the same D2/epilogue source region that the debug post-dequant tap reads.
- Do not fall back to public grouped matmul, scale guessing, or alternate packed-weight interpretation.

Git/worktree state for this handoff:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- Origin: `git@github.com:Zao-0/vllm-ascend-MoE-svdq.git`
- Pre-existing dirty/untracked paths still intentionally left out:
  - `csrc/utils/inc/kernel/moe_distribute_base.h`
  - `csrc/build_out/`
  - `extra-info/`

## Stage 2.2 Appendix GMM2 Official-Path Requirements - 2026-06-26T11:20Z

Status: IN PROGRESS. This section applies the binding requirements from `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`.

Device boundary for the next attempt:

- Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 npu-smi info`
- Result: physical devices 0, 1, 2, 3 visible; all `910B4`, health `OK`; no running NPU processes.
- Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_raw_c2_device_boundary_npu_smi.log`
- Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python - <<'PY' ... torch_npu device count ...`
- Result: `npu_available: True`, `visible_device_count: 4`, logical devices 0-3 all `Ascend910B4`.
- Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_raw_c2_device_boundary_torch_npu.log`

Official source locations inspected for the GMM2 lifecycle table:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:214-225`: GMM2-only debug initializer stores external packed hidden, scale, expert-token inputs and sets `gmm2OnlyFromPacked_`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:247-336`: official layouts, tile shapes, `BlockMmad`, `BlockEpilogue2`, `layoutD2`, and kernel params.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:245-272`: official AIC runs `GMM1` then `GMM2`; official AIV runs `DispatchAndCombine`; GMM2-only debug diverts both task roles.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:276-315`: workspace/global buffer binding for `cumsumMM`, `gmA2I4_I8`, `gmC2`, `gmPerTokenScale2`, `tokenPerExpert`, and `preSumBeforeRank`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:545-572`: `GetCumsumForMMAIV`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:684-779`: official `GMM2(params)` AIC path and final `BlockMmad::Finalize`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:782-859`: current GMM2-only state seeding and direct V2C signaling.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1078-1234`: successful official `DispatchAndCombine` state construction, hidden pack producer, V2C signaling, and `CombineV2` call.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1270-1345`: `CombineV2` waits C2V flags and invokes `BlockEpilogue2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1351-1410`: workspace offset ordering through `ptrC2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp:140-210`, `:234-250`, `:432-465`: official W4A8 MMAD input copy, L0C-to-GM/Fixpipe store, and C2V flag finalize.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:144-240`: `BlockEpilogue2` reads high/low C2, combines raw C2, applies aux and per-token hidden scale, writes FP32 debug tap.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp:50-80`, `:83-127`, `:245-264`: host attributes, shape/listLen/expertPerRank, and workspace sizing.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/svdqw4_a8_gmm2_debug_readback_def.cpp:96-150`: debug-op external packed hidden, hidden scale, external expert-token inputs, and `swigluLimit` sentinel attribute.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_gmm2_debug_readback.cpp:33-56`: debug ACLNN wrapper forwards external hidden/scale/token tensors and forces MTE HCCL server.

Official-vs-debug state table before the next Gate B probe:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | `BlockEpilogue1` in `DispatchAndCombine` writes high/low packed hidden via `gmA2I4_I8[gmOffsetD]` (`kernel.hpp:1198-1203`). | `GMM2` reads `gmA2I4[gmGroupOffsetA + gmOffsetA]` (`kernel.hpp:759-760`). | Official path follows routing/cumsum setup and `BlockEpilogue1` per sync group (`kernel.hpp:1078-1209`). | `workspaceInfo.ptrA2Int4`; bound at `kernel.hpp:293-294`; offset formula uses `layoutD1.GetOffset(offsetC)` for producer and `layoutA2.GetOffset(offsetA)` plus `gmGroupOffsetA` for consumer. | Official `layoutD1{maxOutputSize,k2}` and `layoutA2{m,k2}` (`dispatch_ffn_combine_w4_a8.h:307-312`). | V2C protects GMM2 start after hidden producer. | `DispatchAndCombine` sets `SYNCFLAGV2C` after each `BlockEpilogue1` sync group (`kernel.hpp:1206-1209`). | `GMM2` waits `SYNCFLAGV2C` at group 0 and sync groups (`kernel.hpp:736-738`). | Full path continues to `CombineV2`, then reset/shmem cleanup (`kernel.hpp:1234-1238`). | Debug copies validated external packed hidden into `gmA2I4_I8` on AIV core 0 (`kernel.hpp:814-823`, `:853-858`) and signals V2C directly for all sync groups (`kernel.hpp:782-787`). Deviation: skips official `BlockEpilogue1` producer loop/dequantSum lifecycle. |
| hidden scale `gmPerTokenScale2` | `BlockEpilogue1` writes per-token hidden scale (`kernel.hpp:1198-1203`). | `BlockEpilogue2` reads `gmPerTokenScale` at `gmScaleOffset` (`block_epilogue...v2.hpp:221-227`). | Officially created with hidden pack producer after `dequantSum` setup. | `workspaceInfo.ptrPerTokenScale2`; bound at `kernel.hpp:306-307`; workspace offset at `kernel.hpp:1391-1394`. | Vector length `maxOutputSize`; scale index `(preSrcExpertSum + blockCoord.m()) / 2 + row`. | Same V2C/C2V chain indirectly protects producer and later epilogue. | Producer completes before V2C signal (`kernel.hpp:1206-1209`). | AIC waits V2C before GMM2; AIV waits C2V before `BlockEpilogue2`. | `BlockEpilogue2` applies scale after raw high/low combine. | Debug copies external hidden scale to `gmPerTokenScale2` (`kernel.hpp:821-823`). Deviation: scale is not produced by in-kernel `BlockEpilogue1`, but Stage 2.1 validated it externally. |
| `tokenPerExpert` | Official all-gather/routing state in `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2` (`kernel.hpp:1078`). | `GetCumsumForMMAIV`, `GMM2`, `CombineV2`, and `BlockEpilogue2` token pointer. | Official after routing and cross-rank sync (`kernel.hpp:1078-1082`). | Shmem peer token region: `shmem() + peermemInfo.offsetPeerTokenPerExpert`, bound at `kernel.hpp:309-313`. | `Layout3D(paddedExpertNumAligned, expertPerRank)` (`kernel.hpp:312-313`). | Official cumsum completion also gates GMM1 and later hidden production. | Official GMM1 flag at `kernel.hpp:1098-1100`; V2C after hidden producer. | GMM1 waits at `kernel.hpp:591-594`; GMM2 waits V2C. | Reset at end of full `DispatchAndCombine` (`kernel.hpp:1234-1237`). | Debug copies `externalExpertTokenNums` into local rank slice only (`kernel.hpp:803-809`). Deviation: bypasses official all-gather/pre-sum population for multi-EP; EP=1 probe currently reduces impact but table remains required. |
| `cumsumMM` | `GetCumsumForMMAIV(tokenPerExpert, cumsumMM, ...)` (`kernel.hpp:1080-1082`). | `GMM2` currentM (`kernel.hpp:704-713`) and `CombineV2` currentExpertM (`kernel.hpp:1288-1299`). | Official after token all-gather. | `workspaceInfo.ptrcumsumMM`, bound at `kernel.hpp:284`; workspace offset at `kernel.hpp:1383-1386`. | EP by expertPerRank rows copied with padding in `GetCumsumForMMAIV` (`kernel.hpp:554-571`). | Protected by local syncs and later V2C/C2V usage. | Official before GMM1/GMM2 flows. | GMM2 and CombineV2 read directly after their waits/syncs. | Not separately drained; consumed by loops. | Debug recomputes from copied external tokens (`kernel.hpp:803-810`). Deviation: not produced by official cross-rank all-gather path. |
| `preSumBeforeRank` | Official cross-rank sync helper populates prefix (`kernel.hpp:1078`, reads at `:1084-1089`). | `BlockEpilogue2` receives it (`kernel.hpp:1333-1334`). | Official during cross-rank token exchange. | `workspaceInfo.ptrSumBeforeRank`, bound at `kernel.hpp:314`; field offset after debug regions (`kernel.hpp:1365-1366` plus later workspace setup). | EP by expertPerRank int32. | Protected by official routing/cross-rank sync. | Before hidden producer and epilogue. | AIV epilogue consumes after C2V wait. | Full cleanup after combine. | Debug zeroes it (`kernel.hpp:792-810`). Deviation: correct only for EP=1/rank0; not an official initialization equivalent for multi-EP. |
| GMM2 AIC input tile state | Official `GMM2` creates `layoutA`, `layoutB2`, `layoutScale`, `layoutC` from per-group shape (`kernel.hpp:724-731`). | `BlockMmad` tile copy and MMAD (`block_mmad_w4a4.hpp:140-210`). | Official after V2C wait per sync group. | A: `gmA2I4[gmGroupOffsetA+gmOffsetA]`; B: `gmB2[gmGroupOffsetB+gmOffsetB]`; scale: `gmS2[gmOffsetS]`; C2: `gmC2[gmGroupOffsetC+gmOffsetC]`. | `L1TileShape<128,256,1024>`, `L0TileShape<128,256,256>` (`dispatch_ffn_combine_w4_a8.h:257-283`). | V2C wait before entering group work. | Official AIV producer signals V2C after hidden pack per sync group. | `GMM2` waits at `kernel.hpp:736-738`. | `blockMmad.SynchronizeBlock()` and `Finalize(expertPerRank-1,0)` (`kernel.hpp:775-778`). | Debug calls the same `GMM2(params)` after direct V2C signaling (`kernel.hpp:827-859`). Deviation: surrounding official hidden-producer loop state and `dequantSum` are absent. |
| GMM2 accumulator / D2 region | `BlockMmad` Fixpipe copies L0C to `gmC2` (`block_mmad_w4a4.hpp:432-459`). | `BlockEpilogue2` reads high/low halves from `gmC2` (`block_epilogue...v2.hpp:154-181`). | C2 workspace bound at `kernel.hpp:302`; workspace offset at `kernel.hpp:1405-1410`. | `workspaceInfo.ptrC2`; `gmCOffsetH/L = preSrcExpertSum*n2 + blockCoord.m()*n2 + blockCoord.n() (+ n2)` (`block_epilogue...v2.hpp:154-157`). | `LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n())`; W4A8 stores high/low rows with doubled C allocation. | C2V flags from `BlockMmad::Finalize`. | `Finalize` called from `BlockMmad` after final K tile and again at GMM2 final drain (`block_mmad_w4a4.hpp:464-465`, `kernel.hpp:775-778`). | `CombineV2` waits flags before epilogue (`kernel.hpp:1321-1324`, `:1343-1345`). | `SynchronizeBlock` plus `Finalize`. | Debug still uses official `gmC2`; raw-C2 diagnostic taps the C2 high/low combine before scale (`block_epilogue...v2.hpp:193-207`). Gate B not yet launched after raw-C2 patch. |
| C2V handoff state | `BlockMmad::Finalize(syncLoopIdx, flag)` sets cross-core flags (`block_mmad_w4a4.hpp:234-250`, `:464-465`). | `CombineV2` waits cross-core flags (`kernel.hpp:1321-1324`). | `syncLoopIdx` is set to groupIdx for last loop of each group (`kernel.hpp:739-742`). | Cross-core flag namespace, hard-sync path uses flag id `syncGroupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT + flag`. | Per expert/sync group. | C2V. | After Fixpipe L0C->GM for final K tile. | Before each `BlockEpilogue2` tile and final drain. | `GMM2` finalizes through last expert (`kernel.hpp:775-778`). | Debug relies on same `GMM2` finalize. Deviation under investigation: whether approximate V2C/token state causes GMM2 to skip active tiles or write a different C2 extent. |
| `BlockEpilogue2` input state | Official `CombineV2` passes `gmCGMM2`, `gmC2`, `gmPerTokenScale2`, `ptrMAux2`, tile coord/shape, groupIdx, `preSrcExpertSum`, `preSumBeforeRank`, `listLen` (`kernel.hpp:1333-1334`). | `BlockEpilogue2` combines high/low C2, adds aux, multiplies hidden scale, writes debug FP32 and BF16 output. | Epilogue params official at `kernel.hpp:1168-1180`; debug at `kernel.hpp:839-852`. | `gmCGMM2` debug output offset `(preSrcExpertSum*n2 + blockCoord.m()*n2)/2 + blockCoord.n()` (`block_epilogue...v2.hpp:167-169`). | AIV splits m rows by subcore in 32-row chunks (`kernel.hpp:1309-1335`). | Waits C2V before epilogue. | C2V from GMM2. | `CombineV2` waits before tile loop. | `BlockEpilogue2::Finalize` if any plus CombineV2 final waits. | Debug uses same `CombineV2` and `BlockEpilogue2`; raw mode returns before aux/scale/BF16 to expose Gate B. Deviation: sentinel `swigluLimit` toggles diagnostic behavior only in debug path. |
| FP32 post-dequant debug tap | Official `BlockEpilogue2` writes FP32 post-dequant under `W4A8_DEBUG` after aux and hidden scale (`block_epilogue...v2.hpp:221-235`). | Host probe reads `gmm2PostDequant` output. | Official debug tap in `BlockEpilogue2`. | Output tensor `params.ptrDebugGMM2`; passed to `gmCGMM2`; op output defined as `gmm2PostDequant` (`svdqw4_a8_gmm2_debug_readback_def.cpp:137-144`). | Active rows by n2, with offset divided by 2 for W4A8 high/low pair. | MTE3 event `EVENT_ID7`. | After V pipe scale. | Output read after stream completion. | Event toggles around copy. | Normal debug path has all-zero Gate C. Raw diagnostic writes pre-scale high/low combine to same output (`block_epilogue...v2.hpp:200-207`) and is intended as Gate B evidence. |

Required status fields after the raw-C2 Stage 2.2 probe:

| Field | Current value after 2026-06-26T11:25Z raw-C2 launch |
|---|---|
| `official_gmm2_entry_reached` | True: `official_gmm2_kernel_launched: true` in the raw-C2 and normal summaries. |
| `official_gmm2_loop_count` | Missing; must be added or inferred from stronger device evidence. |
| `official_gmm2_active_tile_count` | Not verified; raw-C2 readback stayed zero, so active C2 tiles are not proven. |
| `official_gmm2_aic_raw_output_finite` | True: raw-C2 sentinel output is finite. |
| `official_gmm2_aic_raw_output_nonzero` | False: raw-C2 sentinel output is all zeros (`max_abs: 0.0`, `mean_abs: 0.0`, `nonzero: false`). |
| `official_gmm2_aic_reference_passed` | False: raw-C2 probe summary `passed: false`; no accumulator oracle passed. |
| `official_gmm2_c2v_handoff_verified` | False: the post-C2V raw readback did not expose nonzero C2 data. |
| `official_gmm2_post_dequant_finite` | True. |
| `official_gmm2_post_dequant_nonzero` | False. |
| `official_gmm2_post_dequant_reference_passed` | False. |
| `official_gmm2_numerical_gate_passed` | False. |

Raw-C2 and normal-path validation after rebuild/install:

- Rebuilt and installed `ops_aclnn`, `cust_opapi`, custom OPP metadata, repo-local custom OPP, system OPP, and `vllm_ascend_C`.
- ABI validation passed:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`
  - Result: `1 passed, 16 warnings`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_pytest_abi.log`
- Installed custom-op source surface contains `svdqw4_a8_gmm2_debug_readback.cpp` under the official `dispatch_ffn_combine_w4_a8` AscendC implementation directory. The narrow dynamic-file lookup returned no matching dynamic Python file.
- Raw-C2 sentinel probe:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --swiglu-limit 424242 --summary-name phase_stage2_gmm2_raw_c2_probe.json`
  - Result: probe summary `passed: false`, `skipped: false`; raw readback output shape `[128, 2048]`, finite `true`, nonzero `false`, `max_abs: 0.0`, `mean_abs: 0.0`.
  - Healthy input boundary in the same summary: canonical hidden BF16 finite/nonzero (`max_abs: 4.3125`), hidden INT8 finite/nonzero (`max_abs: 127.0`), hidden INT4 packed finite/nonzero (`max_abs: 128.0`), hidden scale finite/nonzero (`max_abs: 0.03395669162273407`), packed hidden exact match `true`, mismatch count `0`, W2 packed weight and W2 scale finite/nonzero, public grouped matmul not used, production SVDQ host tiling fail-closed.
  - Reference mismatch remains: actual rows are zeros while the unfused reference is nonzero over 64 compared rows (`max_abs: 0.4712103307247162`, `mean_abs: 0.060490094125270844`).
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_probe.log`
  - Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_probe.json`
- Normal non-sentinel probe after the same rebuild/install:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --summary-name phase_stage2_gmm2_raw_c2_normal_check.json`
  - Result: `passed: false`, `skipped: false`; normal post-dequant readback remains finite all-zero and does not match the nonzero unfused reference.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_normal_check.log`
  - Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_normal_check.json`

Interpretation:

- Gate A remains PASS for the mixed-hidden boundary: the packed hidden and scale inputs supplied to the official-path debug op are deterministic, finite/nonzero, and packed hidden matches the software reference exactly.
- Gate B is FAIL: the raw-C2 sentinel did not expose nonzero official GMM2 accumulator data after the C2V wait.
- Gate C is FAIL: the normal post-dequant readback remains finite all-zero against a nonzero real-checkpoint unfused reference.
- Under the appendix requirements, no additional behavioral kernel patch was made after this result. The next source change must choose a specific deviation from the official-vs-debug state table and tie it to an official counterpart.

Permitted next attempt:

- The already-committed raw-C2 diagnostic is allowed only as a Gate B readback attempt because it reads the official `gmC2` high/low source inside `BlockEpilogue2` after the official `CombineV2` C2V wait.
- No further behavioral patch is permitted until a specific deviation from the table above is selected and tied to an official source counterpart.
- Generated custom-op install mirrors under `vllm_ascend/_cann_ops_custom` were refreshed for local validation but are not intended to be committed.

## Current Handoff State - 2026-06-26T10:32Z

Status: IN PROGRESS. Stage 2.2 remains open.

Authoritative baseline:

- `svdq_qwen35_moe_clean_implementation_report_clarified_completed.md` remains the accepted baseline.
- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` is the active working prompt.
- The current unresolved boundary is still: SVDQ-modified canonical BF16 hidden -> official hidden quant/pack -> official W4A8 GMM2 AIC -> official AIV dequant/readback.
- Production `DispatchFFNCombineW4A8SVDQ` host tiling remains fail-closed.
- Public `torch_npu.npu_grouped_matmul` was not used.

Visible device boundary:

- Environment: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Physical devices visible through `npu-smi`: 0, 1, 2, 3 only; all are `910B4`, healthy, and no NPU processes were running at boundary capture time.
- Torch/NPU runtime: `npu_available: True`, `visible_device_count: 4`, logical devices 0-3 all report `Ascend910B4`.
- Evidence:
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_device_boundary_npu_smi.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_device_boundary_torch_npu.log`

Files changed in this handoff:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - Added `SeedGMM2OnlyTokenState`.
  - Added `SeedGMM2OnlyPackedHidden`.
  - Updated isolated `GMM2OnlyFromPacked` so AIC seeds only token/cumsum state before entering official `GMM2(params)`.
  - Updated isolated `GMM2OnlyDequantReadback` so AIV seeds packed hidden, hidden scale, and token/cumsum state, synchronizes, signals `SYNCFLAGV2C`, then enters official `CombineV2(params, blockEpilogue2)`.
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
  - Tracked copy of this handoff section.

Behavior reused from official code:

- Official W4A8 GMM2 AIC path remains `GMM2(params)`.
- Official AIV post-dequant path remains `CombineV2(params, blockEpilogue2)` with `BlockEpilogue2`.
- Official packed hidden GM buffer remains `gmA2I4_I8`.
- Official hidden per-token scale buffer remains `gmPerTokenScale2`.
- Official expert-token/cumsum state remains `tokenPerExpert`, `cumsumMM`, and `preSumBeforeRank`.

SVDQ-specific/debug-only adaptation:

- The isolated GMM2-only debug operator receives externally validated packed hidden INT4 and hidden scale from the mixed hidden boundary and copies them into the official internal buffers.
- The AIV side now owns the packed-hidden/hidden-scale producer role before signalling V2C, matching the official producer -> AIC consumer -> C2V -> AIV dequant ownership direction more closely than the prior AIC-seeded path.
- AIC still seeds token/cumsum state before `GMM2` because the isolated `GMM2` loop reads cumsum-derived state before the V2C wait.

Build/install evidence after the AIV-producer fix:

- Focused debug kernel build:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`
  - Result: pass, `[100%] Built target svdqw4_a8_gmm2_debug_readback_ascend910b`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_kernel_build_aiv_producer_fix.log`
- `ops_aclnn` rebuild:
  - Result: pass, `[100%] Built target ops_aclnn`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_rebuild_ops_aclnn_aiv_producer_fix.log`
- `cust_opapi` rebuild:
  - Result: pass, `[100%] Built target cust_opapi`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_rebuild_cust_opapi_aiv_producer_fix.log`
- Ascend custom-op metadata regeneration:
  - Result: pass, exit 0.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_ascendc_ops_config_aiv_producer_fix.log`
- CMake custom-op package install staging:
  - Result: pass, exit 0.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_cmake_install_aiv_producer_fix.log`
- Repo-local custom OPP install:
  - Result: pass, `SUCCESS`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_install_repo_custom_ops_aiv_producer_fix.log`
- System OPP install:
  - Result: pass, `SUCCESS`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_install_system_opp_aiv_producer_fix.log`
- In-place Python extension rebuild:
  - Result: pass, `[100%] Built target vllm_ascend_C`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_rebuild_vllm_ascend_C_aiv_producer_fix.log`
- In-place Python extension install:
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_install_vllm_ascend_C_aiv_producer_fix.log`
- ABI registration test:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`
  - Result: pass, `1 passed, 16 warnings`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_pytest_aiv_producer_fix.log`

Stage 2.2 real-device probe after the AIV-producer fix:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --summary-name phase_stage2_gmm2_from_mixed_hidden_aiv_producer_fix.json`
- Result: fail, `passed: false`.
- Evidence:
  - Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_from_mixed_hidden_probe_aiv_producer_fix.log`
  - Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_from_mixed_hidden_aiv_producer_fix.json`

Validated healthy inputs and invariants:

- `torch_op_registered: true`
- `official_gmm2_kernel_launched: true`
- `public_grouped_matmul_used: false`
- `production_svdq_host_tiling_fail_closed: true`
- HCCL group: initialized by probe, backend `hccl`, rank 0, world size 1, logical device 0.
- Real-checkpoint layer: `model.language_model.layers.0.mlp.experts`.
- Shape: 8 local experts, hidden size 2048, intermediate size 512, 16 tokens, top-k 8, active rows 128.
- Expert token counts: `[16, 16, 16, 16, 16, 16, 16, 16]`, total 128, expert-contiguous rows true.
- Canonical hidden BF16: finite and nonzero, shape `[128, 512]`, `max_abs: 4.3125`, `mean_abs: 0.3363776206970215`.
- Hidden INT8: finite and nonzero, shape `[128, 512]`, `max_abs: 127.0`, `mean_abs: 14.154754638671875`.
- Hidden packed INT4 debug tensor: finite and nonzero, shape `[128, 512]`, `max_abs: 128.0`, `mean_abs: 42.087799072265625`.
- Hidden scale: finite and nonzero, shape `[128]`, `max_abs: 0.03395669162273407`, `mean_abs: 0.02411513403058052`.
- Hidden packed exact reference:
  - `exact_match: true`
  - `mismatch_count: 0`
  - `max_abs_diff: 0`
  - sample actual/expected bytes both start `[0, 15, -32, -33, -48, 0, 0, -1, -16, 15, -17, -47, 15, -35, 15, -1]`.

Failing boundary:

- GMM2 post-dequant active tensor remains all zeros after the AIV-producer fix:
  - shape `[128, 2048]`
  - dtype `torch.float32`
  - finite: `true`
  - nonzero: `false`
  - `max_abs: 0.0`
  - `mean_abs: 0.0`
  - `nan_count: 0`
  - `inf_count: 0`
- Unfused real-checkpoint reference is nonzero.
- Reference comparison over 64 rows:
  - `max_abs: 0.4712103307247162`
  - `mean_abs: 0.060490094125270844`
  - failed tolerance: `max_abs <= 0.0002`, `mean_abs <= 0.00002`
  - actual sample: eight zeros
  - expected sample starts `[0.051994405686855316, -0.0029550495091825724, 0.02753283828496933, 0.03887496888637543, 0.08114911615848541, -0.051662545651197433, -0.01274419017136097, -0.027047518640756607]`.

Current diagnosis:

- The AIV-producer ownership correction is necessary but not sufficient; it did not change the observed all-zero GMM2 post-dequant failure.
- The packed hidden boundary remains exact and should not be re-debugged as a packing or scale issue.
- The official debug op launches and reaches the same numerical failure, so the remaining bug is likely in the isolated GMM2-only handoff/readback state rather than in checkpoint loading or mixed hidden quantization.
- Next debugging should stay inside the official-kernel-derived GMM2-only path and inspect:
  - C2V flag wait/signal/finalize ordering around official `GMM2` and `CombineV2`;
  - whether AIC writes the same D2/epilogue source region that `BlockEpilogue2` reads in `CombineV2`;
  - `cumsumMM`, `tokenPerExpert`, and `preSumBeforeRank` state as seen by AIC and AIV after the new seeding;
  - whether the isolated debug path needs the official GMM2 pre/post loop state normally established by the preceding GMM1/SwiGLU phase.
- Do not switch to public grouped matmul, scalar W4A8 substitutes, scale guessing, or packed-weight reinterpretation.

Git/worktree state for this handoff:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- Origin: `git@github.com:Zao-0/vllm-ascend-MoE-svdq.git`
- Intended staged files:
  - `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- Pre-existing dirty/untracked paths still intentionally left out:
  - `csrc/utils/inc/kernel/moe_distribute_base.h`
  - `csrc/build_out/`
  - `extra-info/`
- Generated custom-op install mirrors under `vllm_ascend/_cann_ops_custom` were refreshed for validation but are not intended to be committed.

## Current Handoff State - 2026-06-26T11:00Z

Status: IN PROGRESS. Stage 2.2 remains open.

Purpose of this handoff:

- The environment is expected to be rebuilt, so this section records the current source state, evidence, and unresolved diagnostic boundary.
- This is not a Stage 2.2 pass. The new raw-C2 path has build evidence only; it has not yet been installed or launched as a real-device numerical gate.

Authoritative constraints still in force:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` was read and remains the active working prompt.
- The unresolved boundary remains: SVDQ-modified canonical BF16 hidden -> official hidden quant/pack -> official W4A8 GMM2 AIC -> official AIV dequant/readback.
- Production `DispatchFFNCombineW4A8SVDQ` host tiling remains fail-closed.
- Public `torch_npu.npu_grouped_matmul` was not modified, debugged, or used.
- The packed hidden INT4 exact-match gate remains accepted and should not be reopened as a packing/scale issue without new contradictory real-device evidence.

Official full-W4A8 control evidence:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py --require-npu --summary-name phase_stage2_control_full_w4a8_debug_after_aiv_producer.json`
- Result: pass, `passed: true`.
- Evidence:
  - Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_control_full_w4a8_debug_after_aiv_producer.log`
  - Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_control_full_w4a8_debug_after_aiv_producer.json`
- Interpretation:
  - The installed full official W4A8 debug path, including official GMM2 and post-dequant readback, is healthy.
  - The remaining all-zero failure is isolated to the GMM2-only path from SVDQ-modified hidden, not to the official full W4A8 kernel generally.

Files changed in this handoff:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp`
  - Added `BlockEpilogue2::Params::rawDebugOnly`.
  - Added a debug-only branch in `BlockEpilogue2::operator()` after the official high/low C2 accumulator combination (`high * 16 + low`) and before hidden-scale/weight-aux dequantization.
  - In raw mode, the combined raw C2 accumulator is copied to the existing `gmm2PostDequant` debug GM output under `W4A8_DEBUG`, then the epilogue tile returns without applying hidden scale, auxiliary weight, or BF16 final output conversion.
  - The branch waits/releases the already-issued weight-aux MTE2 event before returning so the UB slot lifecycle is not left with a pending event.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - The isolated `GMM2OnlyDequantReadback` path passes `rawDebugOnly = true` only when `swigluLimit` is in the sentinel range `(400000, 500000)`.
  - The intended probe switch is `--swiglu-limit 424242`.
  - Normal probes keep the existing post-dequant path because their default `swigluLimit` is outside the sentinel range.

Why this diagnostic exists:

- The previous Stage 2.2 real-device probe still returned all-zero GMM2 post-dequant output despite healthy canonical hidden, exact packed hidden INT4, nonzero hidden scale, op registration, and kernel launch.
- The raw-C2 branch is designed to distinguish:
  - nonzero raw C2 with zero post-dequant: AIC produced data and the remaining issue is in AIV dequant/scale/readback;
  - zero raw C2: the isolated GMM2-only AIC producer did not write the expected C2 data or the AIV readback is pointed at the wrong region.
- This diagnostic directly reuses the official W4A8 `BlockEpilogue2` entry and official C2 accumulator combine location. It is not a public grouped-matmul experiment, scalar substitute, or alternative dequant formula.

Focused build evidence for the raw-C2 diagnostic:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`
- Result: pass, `[100%] Built target svdqw4_a8_gmm2_debug_readback_ascend910b`.
- Evidence:
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_debug_kernel_build_raw_c2_probe.log`
- Scope:
  - This proves the raw-C2 diagnostic branch compiles through the focused AscendC generated-kernel target.
  - It does not prove numerical correctness and does not close Stage 2.2 because the rebuilt op was not installed/launched after this patch before the environment rebuild handoff.

Next action after environment rebuild:

1. Re-establish the four-NPU boundary with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
2. Rebuild/install the raw-C2 patched GMM2 debug op through the supported flow:
   - `cmake --build csrc/build --target ops_aclnn -- -B -j1`
   - `cmake --build csrc/build --target cust_opapi -- -B -j1`
   - `/usr/local/python3.12.13/bin/python3.12 csrc/cmake/scripts/util/ascendc_ops_config.py -p csrc/build/binary/ascend910b/bin -s ascend910b`
   - `cmake --install csrc/build --prefix /tmp/svdq_install_gmm2_debug_raw_c2_probe`
   - install to both repo-local custom OPP and system OPP as in the prior successful Stage 2.2 install flow.
3. Rebuild/install the in-place Python extension if adapter or linked op API freshness is uncertain.
4. Run the raw-C2 sentinel probe:
   `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --swiglu-limit 424242 --summary-name phase_stage2_gmm2_raw_c2_probe.json`
5. Inspect `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_probe.json`, especially `stage.gmm2.post_dequant_active.nonzero`, `max_abs`, and `mean_abs`.
6. Run the normal probe again without the sentinel after the raw-C2 result to verify the ordinary post-dequant path has not regressed.

Git/worktree state for this handoff:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- Origin: `git@github.com:Zao-0/vllm-ascend-MoE-svdq.git`
- Previous pushed commit before this diagnostic: `906fcd4ca2f74b398a8aeb1a5606572c112cfd90` (`Record Stage 2.2 AIV producer handoff`).
- Intended committed files for this handoff:
  - `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp`
  - `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- Pre-existing dirty/untracked paths still intentionally left out:
  - `csrc/utils/inc/kernel/moe_distribute_base.h`
  - `csrc/build_out/`
  - `extra-info/`
