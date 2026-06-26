# SVDQ Qwen3.5 MoE Clean Implementation - Stage 2 Report

## Baseline - 2026-06-26T06:17Z

Status: IN PROGRESS.

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
