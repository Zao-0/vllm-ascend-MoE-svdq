# SVDQ Qwen3.5 MoE Clean Implementation - Stage 2 Report

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | PASS | Fresh real-device recheck with Appendix GMM2 official-path fields passes Gate A/B/C. |
| Stage 2.3 same-routing and final-combine isolated gates | PASS | Existing isolated real-checkpoint Stage 2.3 evidence is unblocked by the fresh Stage 2.2 pass. |
| Stage 2.4 production/four-NPU admission | IN PROGRESS | Production residual W4A8 GMM execution and four-NPU target-model E2E validation remain open. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | Production enable remains false and host tiling must remain fail-closed. |

## Stage 2.4 Official W4A8 Wrapper Type Bound - 2026-06-27

Purpose:

- The SVDQ production kernel now includes the official
  `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h` wrapper header instead of directly
  including `moe_init_routing_quant_v2.cpp`.
- `SVDQOfficialW4A8Op` aliases
  `DispatchFFNCombineW4A8Impl::DispatchFFNCombineW4A8<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true>`, the same official
  W4A8 specialization used for the packed-W4 path.
- `OfficialW4A8WrapperTypeBound()` records type-level binding to the official wrapper without constructing or launching
  it. Production residual W4A8 GMM execution remains disabled because the SVDQ kernel still has no official
  `op.Init(...)` or `op.Process()` invocation.

Appendix GMM2 official-path constraints accepted for subsequent behavioral work:

- No public `torch_npu.npu_grouped_matmul` work, scalar W4A8 GEMM, host weight unpacking, scale guessing, or custom
  packed-W4 reinterpretation is allowed.
- The official W4A8 AIC/AIV lifecycle remains the only source of truth for GMM1, GMM2, packed-weight access, tiling,
  accumulator/Fixpipe state, C2V handoff, `BlockEpilogue2`, and `CombineV2`.
- This edit is not a behavioral GMM2 patch. It records the wrapper type boundary only; the next behavioral correction
  must first document the official-vs-debug state deviation required by the Appendix GMM2 official-path table.

Official source locations inspected for this attempt:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h`
- `csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h`
- `tools/svdq_kernel_contract_manifest.py`
- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`

Machine-checkable source constraints:

- `source_proof.kernel_residual_gmm_official_full_lifecycle_call_surface_recorded=true`
- `production_fail_closed.residual_gmm_official_wrapper_type_bound=true`
- `source_proof.kernel_residual_gmm_official_full_lifecycle_execution_enabled=false`
- `production_fail_closed.residual_gmm_execution_enabled=false`
- `production_admission.production_enable_allowed=false`

Validation for this edit:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`

## Stage 2.4 Embedded Official W4A8 Tiling Pointer Resolved - 2026-06-27

Purpose:

- The production SVDQ kernel now derives the GM address of the embedded official
  `DispatchFFNCombineW4A8TilingData` from the SVDQ `tilingGM` payload:
  `residualW4A8Bridge.officialTiling`.
- The full-lifecycle descriptor now carries this embedded official tiling pointer instead of a null placeholder.
- Residual W4A8 GMM execution remains disabled because the production kernel still does not instantiate and invoke the
  official `DispatchFFNCombineW4A8` wrapper. The next boundary is the actual wrapper call plus fused production
  numerical validation.

Machine-checkable source constraints:

- `source_proof.kernel_residual_gmm_official_full_lifecycle_call_surface_recorded=true`
- `production_fail_closed.residual_gmm_embedded_official_tiling_pointer_recorded=true`
- `source_proof.kernel_residual_gmm_official_full_lifecycle_execution_enabled=false`
- `production_fail_closed.residual_gmm_execution_enabled=false`
- `production_admission.remaining_execution_requirements.residual_w4a8_gmm_execution_enabled=false`
- `production_admission.production_enable_allowed=false`

Validation for this edit:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

## Stage 2.4 Official W4A8 Full-Lifecycle Call Surface Recorded - 2026-06-27

Purpose:

- Added a production SVDQ kernel descriptor for the exact official `DispatchFFNCombineW4A8` full-lifecycle call
  surface: `x`, residual `w1/w2`, residual scales/biases, `expertId`, `probs`, `xActiveMask`, `out`,
  `expertTokenNums`, workspace, the SVDQ tiling pointer, and a future embedded official W4A8 tiling pointer.
- `runtime_.tiling` now preserves the production SVDQ `tilingGM` pointer so the eventual official-wrapper bridge can
  derive or pass the correct official tiling data instead of reconstructing state from names.
- The official lifecycle readiness check deliberately requires `launch.officialTiling != nullptr`; today that pointer is
  still null, so residual W4A8 GMM execution remains disabled. This records the next concrete implementation boundary:
  pass an embedded official `DispatchFFNCombineW4A8TilingData` GM pointer to the official wrapper and validate fused
  production numerics.

Machine-checkable source constraints:

- `source_proof.kernel_residual_gmm_official_full_lifecycle_call_surface_recorded=true`
- `source_proof.kernel_residual_gmm_official_full_lifecycle_execution_enabled=false`
- `production_fail_closed.residual_gmm_official_full_lifecycle_call_surface_recorded=true`
- `production_fail_closed.residual_gmm_official_full_lifecycle_execution_enabled=false`
- `production_fail_closed.residual_gmm_execution_enabled=false`
- `production_admission.remaining_execution_requirements.residual_w4a8_gmm_execution_enabled=false`
- `production_admission.production_enable_allowed=false`

Validation for this edit:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

## Stage 2.4 Host Tiling Metadata Build Before Fail-Closed Return - 2026-06-27

Purpose:

- Production host tiling now runs shape/dtype/platform validation, builds the workspace map, sync table, dispatch
  routing tiling, official residual W4A8 tiling bridge, BF16/residual/mixed/final-combine shape tables, and workspace
  size metadata before the final fail-closed return.
- The production tiling function still returns `ge::GRAPH_FAILED`; it does not admit a production launch and does not
  claim residual W4A8 GMM execution.
- This corrects the previous source state where the fail-closed return happened before the official W4A8 bridge build,
  leaving the production metadata construction code unreachable.

Machine-checkable source constraints:

- `source_proof.host_tiling_fail_closed_after_metadata_construction=true`
- `production_fail_closed.host_tiling_metadata_builds_before_fail_closed=true`
- `production_fail_closed.host_tiling_returns_graph_failed=true`
- `production_fail_closed.host_tiling_success_enabled=false`
- `production_fail_closed.residual_gmm_execution_enabled=false`
- `production_admission.production_enable_allowed=false`

Validation for this edit:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

## Stage 2.4 Residual W4A8 Official Tiling Bridge Consumed - 2026-06-27

Purpose:

- This is a source-boundary/prod-admission bookkeeping step only. It is not numerical progress under the Appendix GMM2
  official-path gate and does not enable production residual W4A8 GMM execution.
- Added kernel-side consumption of the embedded official W4A8 tiling bridge through `ResidualW4A8BridgeTiling()` and
  `ResidualGmmOfficialTilingBridgeReady(uint32_t stageId)`.
- `RunResidualGmmStage` now requires the tiling-aware official bridge readiness check, then still returns `false`.
  Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed until the official W4A8 AIC/AIV lifecycle is directly
  reused and production real-device numerical gates pass.

Machine-checkable source constraints:

- The kernel validates embedded official shape fields: `M=m`, `K=hiddenSize`, `N=2*intermediateSize`,
  `listLen=expertPerRank`, nonzero official workspace bytes, and `hostExecutionFailClosed=true`.
- The kernel validates official W4A8 tiling fields: `isTransposeB=false`, `isWeightNz=true`, matching `topK`,
  `worldSize`, `maxOutputSize`, `swigluLimit`, CoC tile constants, and the routed quant tiling key.
- The residual stage mapping remains explicit:
  GMM1 uses `launch.k == officialK` and `launch.n == officialN`; GMM2 uses `launch.k == officialN/2` and
  `launch.n == officialK`.
- Scalar/public grouped-matmul execution remains absent. The public `torch_npu.npu_grouped_matmul` path was not used.

Expected manifest state after regeneration:

- `source_proof.kernel_residual_gmm_official_tiling_bridge_consumed=true`
- `production_fail_closed.residual_gmm_official_tiling_bridge_consumed=true`
- `production_fail_closed.residual_gmm_execution_enabled=false`
- `production_admission.remaining_execution_requirements.residual_w4a8_gmm_execution_enabled=false`
- `production_admission.production_enable_allowed=false`

Validation for this edit:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
- `git diff --check -- csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

## Stage 2.2 Appendix-Field Real-Device Recheck Passed - 2026-06-27T04:58Z

Latest authoritative status:

- Stage 2.2 is now passed under `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`.
- The probe ran on logical NPU 0 with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`; `npu-smi` reported physical NPUs 0-3 as
  910B4 devices with no active NPU processes, and `torch.npu.device_count()` reported exactly 4 visible devices.
- The public `torch_npu.npu_grouped_matmul` path was not used.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

Device command:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} \
python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py \
  --require-npu \
  --device-id 0 \
  --top-k 1 \
  --route-experts 0 \
  --local-num-experts 8 \
  --num-tokens 16 \
  --max-output-size 512 \
  --gmm2-reference-max-rows 64 \
  --summary-name stage2/phase_stage2_gmm2_current_recheck_top1_expert0.json
```

Artifacts:

- Device log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_current_recheck_appendix_fields.log`
- Device exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_current_recheck_appendix_fields.exitcode`
- Stage 2.2 summary:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_gmm2_current_recheck_top1_expert0.json`
- Manifest log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_4_manifest_after_appendix_field_probe.log`
- Production admission manifest:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

Stage 2.2 result:

- Probe summary: `passed=true`, `skipped=false`.
- `stage.passed=true`.
- Appendix fields are all true:
  `official_vs_debug_state_table_complete`,
  `gate_a_routing_identity_complete`,
  `gate_a_prefix_and_padded_row_evidence_complete`,
  `gate_b_exact_aic_raw_or_d2_boundary_identified`,
  `gate_b_aic_raw_output_reference_passed`,
  `gate_c_validated_after_gate_b_nonzero`, and
  `active_failure_all_zero_post_dequant_resolved`.
- Required numerical flags are true:
  `official_gmm2_aic_raw_output_nonzero`,
  `official_gmm2_aic_reference_passed`,
  `official_gmm2_post_dequant_nonzero`,
  `official_gmm2_post_dequant_reference_passed`, and
  `official_gmm2_numerical_gate_passed`.

Regenerated manifest result:

- `production_admission.stage2_2_official_gmm2_gate.status=passed`
- `production_admission.stage2_2_official_gmm2_gate.evidence_status=current_recheck_passed`
- `production_admission.stage2_2_official_gmm2_gate_passed=true`
- `production_admission.stage2_3_real_checkpoint_composition_gate.status=passed`
- `production_admission.stage2_3_isolated_gate_passed=true`
- `production_admission.production_enable_allowed=false`
- Remaining requirements:
  `residual_w4a8_gmm_execution_enabled=false` and `four_npu_target_model_e2e_validated=false`.

Next unresolved boundary:

- Do not enable production host tiling yet. The next required work is wiring the residual W4A8 GMM production path
  through the official W4A8 AIC/AIV lifecycle, then validating fused production numerics and exactly-four-NPU target
  model E2E.

## Stage 2.2 Authoritative Status Reset - 2026-06-27

Historical section superseded by the 2026-06-27T04:58Z Appendix-field real-device recheck above.

Previous binding requirement:

- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` is now the active constraint for
  Stage 2.2.
- The authoritative state at that time was Stage 2.0 PASS, Stage 2.1 PASS, Stage 2.2 FAIL / IN PROGRESS,
  Stage 2.3+ BLOCKED, and production FAIL-CLOSED.
- The active failure is the official GMM2 post-dequant readback returning all zeros while the real-checkpoint unfused
  official-contract reference is nonzero.
- Do not reopen BF16 producer work or the packed-hidden Stage 2.1 work. The active problem is the official W4A8 GMM2
  producer/consumer lifecycle and state reconstruction.
- Do not make another behavioral GMM2 patch until the official-vs-debug state table below is updated with the
  deviation being corrected.
- Gate A still needs complete routed-row identity, prefix, expert-local row, and padded-row evidence. Gate B must expose
  the exact official AIC raw accumulator/D2 boundary and prove it finite/nonzero/reference-matched. Gate C must validate
  the official `BlockEpilogue2` / `CombineV2` FP32 post-dequant output only after Gate B is nonzero.
- The public `torch_npu.npu_grouped_matmul` path remains forbidden and was not used.
- Production SVDQ host tiling remains fail-closed. Source-boundary manifests, tiling records, compilation, registration,
  and kernel entry do not count as Stage 2.2 numerical progress.

Current code-state note:

- The residual W4A8 official tiling bridge recorded later in this report is a fail-closed source boundary only. It does
  not enable residual W4A8 GMM execution and does not satisfy Stage 2.2 Gate B or Gate C.
- This section is retained to explain the fail-closed reset before fresh Appendix-field evidence was generated.

## Stage 2.2 Appendix Evidence Fields Added to Probe - 2026-06-27

Purpose:

- The Stage 2.2 manifest was already fail-closed unless evidence contained explicit Appendix GMM2 official-path revision
  fields. The real-device probe did not yet emit those fields, so future runs could not distinguish a missing-evidence
  state from an actual Gate A/B/C failure.
- Added `appendix_gmm2_official_path` emission to `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`.
- This is evidence plumbing only. It does not change the official W4A8 kernel, synchronization, C2V/V2C flags,
  packed-W4 access, scale formulas, BF16 producers, or production SVDQ host tiling.

Fields now emitted by each probe mode:

- `official_vs_debug_state_table_complete`
- `gate_a_routing_identity_complete`
- `gate_a_prefix_and_padded_row_evidence_complete`
- `gate_b_exact_aic_raw_or_d2_boundary_identified`
- `gate_b_aic_raw_output_reference_passed`
- `gate_c_validated_after_gate_b_nonzero`
- `active_failure_all_zero_post_dequant_resolved`

Gate behavior implemented by the probe:

- Loop-stats mode can record Gate A context but keeps Gate B and Gate C false.
- Raw-C2/D2 mode can record Gate B source-boundary and reference status, but keeps Gate C false.
- Normal post-dequant mode marks Gate C true only after Gate B is identified, nonzero, and reference-passed, and the
  official `BlockEpilogue2` / `CombineV2` post-dequant output is finite, nonzero, and reference-passed.
- Evidence generated before the fields were added remained blocked. The fresh 2026-06-27T04:58Z run above now carries
  these fields and admits Stage 2.2.

Validation:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py tools/svdq_kernel_contract_manifest.py`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

Next unresolved boundary:

- The field-emission work is complete. The fresh run above has populated the fields from actual device execution.

## Stage 2.2 Historical Official-Path Recheck - 2026-06-27

Latest binding requirement:

- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` is now treated as the current
  constraint for Stage 2.2.
- This section is retained as historical evidence only. It is superseded by the authoritative status reset above and must
  not be used to advance Stage 2.3+ or enable production.
- The appendix reopened Stage 2.2 until the official-vs-debug state table and a current Gate A/B/C recheck are recorded
  against the latest official-path requirements.
- The older root-level Stage 2.2 summary remains historical. The current admission evidence is
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_gmm2_current_recheck_top1_expert0.json`.
- The public `torch_npu.npu_grouped_matmul` path remains forbidden and was not used.
- Production SVDQ host tiling remains fail-closed; passing isolated Stage 2.2 and Stage 2.3 gates does not enable
  production fused execution or four-NPU serving.

Machine-checkable guardrail:

- `tools/svdq_kernel_contract_manifest.py` now emits
  `production_admission.stage2_2_official_gmm2_gate.status=fail_in_progress`,
  `stage2_2_official_gmm2_gate_passed=false`, and `stage2_3_isolated_gate_passed=false` for the currently available
  evidence, because it lacks the new Appendix GMM2 official-path revision fields.
- If the current Stage 2.2 recheck file is missing, invalid, or missing those revision fields, Stage 2.2 remains
  `fail_in_progress` and Stage 2.3 remains blocked.
- `production_enable_allowed=false` and `host_tiling_must_remain_fail_closed=true`.

Validation:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `git diff --check -- tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q` -> `46 passed, 16 warnings`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --device-id 0 --top-k 1 --route-experts 0 --local-num-experts 8 --num-tokens 16 --max-output-size 512 --gmm2-reference-max-rows 64 --summary-name stage2/phase_stage2_gmm2_current_recheck_top1_expert0.json`
  -> summary `passed=true`, `skipped=false`.
- Current Stage 2.2 recheck result:
  `official_gmm2_entry_reached=true`, `official_gmm2_loop_count=8`,
  `official_gmm2_active_tile_count=8`, `official_gmm2_loop_stats_valid=true`,
  `official_gmm2_aic_raw_output_nonzero=true`, `official_gmm2_accumulator_int32_reference_passed=true`,
  `official_gmm2_c2v_handoff_verified=true`, `official_gmm2_post_dequant_nonzero=true`,
  `official_gmm2_post_dequant_reference_passed=true`, and `official_gmm2_numerical_gate_passed=true`.
  The strict post-dequant comparison records `max_abs=0.0` and `mean_abs=0.0`.
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
  regenerated the manifest with `stage2_2_official_gmm2_gate.status=fail_in_progress`,
  `evidence_status=current_recheck_missing_appendix_gmm2_official_path_revision_fields`,
  `stage2_2_official_gmm2_gate_passed=false`, `stage2_3_isolated_gate_passed=false`, and
  `production_enable_allowed=false`.

Official W4A8 source anchors inspected:

- Top-level official kernel task split: `dispatch_ffn_combine_w4_a8_kernel.hpp:247-276`.
- Official workspace and GM bindings: `dispatch_ffn_combine_w4_a8_kernel.hpp:287-317` and `1530-1630`.
- Official GMM2 AIC path: `dispatch_ffn_combine_w4_a8_kernel.hpp:687-794`.
- Current debug-only packed-hidden seeding helpers: `dispatch_ffn_combine_w4_a8_kernel.hpp:944-986`.
- Current debug-only dequant readback path: `dispatch_ffn_combine_w4_a8_kernel.hpp:990-1016`.
- Full official AIV hidden producer / external overlay boundary: `dispatch_ffn_combine_w4_a8_kernel.hpp:1323-1412`.
- Official GMM2 AIV dequant / C2V consumer: `dispatch_ffn_combine_w4_a8_kernel.hpp:1448-1525`.
- Official `BlockEpilogue2` dequant and peer-output writeback: `block_epilogue_w4a8post_pertoken_v2.hpp:147-312`.

Official-vs-debug state table checkpoint:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | `BlockEpilogue1` writes hidden packed INT4 during the full AIV path; external overlay can replace the same region at `dispatch_ffn_combine_w4_a8_kernel.hpp:1358-1376`. | `GMM2` reads `gmA2I4` at `dispatch_ffn_combine_w4_a8_kernel.hpp:773-776`. | Bound in `initBuffer` at `dispatch_ffn_combine_w4_a8_kernel.hpp:293-301`. | `workspaceInfo.ptrA2Int4`, offset after `ptrA1Int4`, at `dispatch_ffn_combine_w4_a8_kernel.hpp:1596-1602`. | `layoutA2.GetTileLayout`; GMM2 doubles `currentM` for INT4 at `721-723`. | V2C flag `SYNCFLAGV2C`. | Official full path sets V2C after each hidden producer window at `1380-1383`. | GMM2 waits before group work at `745-747`. | `blockMmad.SynchronizeBlock()` and `Finalize()` at `790-793`. | Debug helpers can seed this region directly at `963-972`; new appendix requires preserving the full official lifecycle instead of approximating it. |
| hidden scale `gmPerTokenScale2` | `BlockEpilogue1` writes per-token hidden scale and external overlay can replace it at `1358-1376`. | `BlockEpilogue2` consumes `gmPerTokenScale2` at `1511-1512` and scales rows at `block_epilogue_w4a8post_pertoken_v2.hpp:248-255`. | Bound at `dispatch_ffn_combine_w4_a8_kernel.hpp:307-310`. | `workspaceInfo.ptrPerTokenScale2`, offset after first per-token scale at `1567-1572`. | One FP32 scale per active routed row; Gate A must prove active-row identity. | V2C/C2V lifecycle shared with hidden packed and GMM2 output. | Same hidden producer window as packed hidden. | AIV `CombineV2` waits C2V before dequant at `1499-1502`. | `BlockEpilogue2::Finalize()` at `1525`. | Prior evidence says finite/nonzero; current appendix requires row-by-row proof with prefix and padded-row identity. |
| `tokenPerExpert` | Official dispatch/routing and cross-rank token exchange populate peer token state before GMM2. | GMM2 uses `cumsumMM`; `BlockEpilogue2` reads per-destination counts at `block_epilogue_w4a8post_pertoken_v2.hpp:277-282`. | Bound through shared memory at `dispatch_ffn_combine_w4_a8_kernel.hpp:312-317`. | `shmem() + peermemInfo.offsetPeerTokenPerExpert`; layout `Layout3D(paddedExpertNumAligned, expertPerRank)`. | Expert-major per EP/rank/group. | HCCL/shared-memory sync and V2C/C2V flags. | Official `DispatchAndCombine` lifecycle, not standalone seeding. | GMM2 and CombineV2 wait on official flags. | Reset after combine at `1412-1415`. | Debug seeding copies `externalExpertTokenNums` at `944-958`; this is now only diagnostic unless proven equivalent to official state. |
| `cumsumMM` | Official token-count cumsum construction in the full lifecycle. | GMM2 reads per-expert active rows at `713-723`; `CombineV2` reads the same at `1466-1477`. | Bound at `287`. | `workspaceInfo.ptrcumsumMM`, offset after expanded-row indices at `1561-1564`. | Expert prefix over local rank/EP. | V2C gating to GMM2 and C2V gating to AIV. | Must match official route/count construction. | GMM2 waits V2C at `745-747`; AIV waits C2V at `1499-1502`. | No separate substitute drain allowed. | Debug `GetCumsumForMMAIV` path at `954-958` must not be treated as a passed production lifecycle. |
| `preSumBeforeRank` | Official cross-rank prefix state. | `BlockEpilogue2` uses it for remote peer offsets at `block_epilogue_w4a8post_pertoken_v2.hpp:277-307`. | Bound at `317`. | `workspaceInfo.ptrSumBeforeRank`, offset after debug GMM regions at `1629-1630`. | Per destination EP and local expert. | Official cross-rank synchronization. | Must be produced before dequant peer-output writeback. | Consumed after C2V wait. | Final peer-output writeback is in `BlockEpilogue2`. | Debug helper zeroes it at `933-960`; appendix forbids relying on this without a direct official counterpart. |
| GMM2 AIC input tile state | Official `GMM2` constructs `inGroupProblemShape`, layouts, block scheduler, and offsets at `713-764`. | `BlockMmad` consumes scale, packed hidden, packed W2, and C2 output at `773-776`. | `GMM2` creates `BlockScheduler` and `BlockMmad` at `689-691`. | Inputs are `gmA2I4`, W2 from `GetTensorAddr`, and output `gmC2`. | L1 tile shape with INT4 doubled-M rows. | V2C wait before work. | Hidden producer sets V2C after each dequant window. | `GMM2` waits V2C by group. | `SynchronizeBlock` and `Finalize`. | Debug GMM2-only path bypasses GMM1/BlockEpilogue1 and must be replaced by official lifecycle preservation. |
| GMM2 accumulator / D2 region | `BlockMmad` writes `gmC2` at `773-776`; debug accumulator tap is optional. | `BlockEpilogue2` reads high/low halves from `gmC2` at `block_epilogue_w4a8post_pertoken_v2.hpp:157-184`. | `gmC2` bound at `305`; workspace offset at `1583-1589`. | `workspaceInfo.ptrC2`; debug `ptrCGMM2` can tap post-dequant/FP32. | `layoutC` with high/low row halves. | C2V flag. | AIC sets C2V via BlockMmad lifecycle. | AIV waits C2V in `CombineV2`. | `BlockEpilogue2::Finalize()`. | Mandatory Gate B must read the exact official accumulator/D2 boundary and prove finite/nonzero/reference parity. |
| C2V handoff state | Official AIC `BlockMmad` completion and cross-core flag lifecycle. | `CombineV2` waits before each dequant tile at `1499-1502`. | `SYNCFLAGC2V=9`, `SYNCFLAGV2C=10` at `45-46`. | Cross-core flags, not a GM tensor. | Per expert sync loop. | C2V and V2C. | Official, group-ordered. | Official, group-ordered. | Remaining waits through `syncLoopIdx` drain at `1521-1524`. | New appendix forbids more speculative flag-order edits unless tied to official code. |
| `BlockEpilogue2` input state | `CombineV2` passes `gmCGMM2`, `gmC2`, `gmPerTokenScale2`, W2 aux, tile coords, group, `preSumBeforeRank` at `1511-1512`. | `BlockEpilogue2::operator()` dequantizes and writes peer output. | Constructed at `1325-1339` or debug readback at `996-1009`. | `gmC2`, `gmPerTokenScale2`, `ptrMAux2`, remote peer memory `offsetD`. | `actualBlockShape.m()/2` after high/low merge. | UB events inside `BlockEpilogue2`. | After C2V wait. | Internal MTE/V waits at `block_epilogue_w4a8post_pertoken_v2.hpp:182-190` and `265-270`. | `Finalize()` waits all UB stages at `132-140`. | Gate C must validate this output only after Gate B is nonzero. |
| FP32 post-dequant debug tap | `BlockEpilogue2` debug branch writes `gmTileGMM2` after high/low merge and scale at `block_epilogue_w4a8post_pertoken_v2.hpp:258-262`. | Host probe reads debug GM. | Debug GM is optional through `ptrDebugGMM2`. | `workspaceInfo.ptrCGMM2` or external debug pointer at `1620-1627`. | FP32 row-major active tile. | Debug uses `EVENT_ID7` MTE3/V guard. | During BlockEpilogue2. | Debug waits `EVENT_ID7`. | `Finalize()` waits debug event under `W4A8_DEBUG`. | Prior all-zero readbacks are insufficient; Gate B and Gate C must be separated and interpreted as specified by the appendix. |

## Stage 2.4 Residual W4A8 Official Bridge Contract - 2026-06-27

Latest code state:

- Added `SVDQResidualGmmOfficialBridgeContract` to `dispatch_ffn_combine_w4_a8_svdq.h`.
- The bridge records the only acceptable production residual-GMM source boundary:
  official `dispatch_ffn_combine_w4_a8`, official AIC GMM producer, official AIV dequant consumer, packed W4 weights,
  official accumulator lifecycle, official C2V handoff, and BF16 residual output into the SVDQ accumulator regions.
- Stage 1 maps `SVDQ_REGION_X_Q` plus `SVDQ_REGION_X_SCALE` to `SVDQ_REGION_ACCUMULATOR_1`.
- Stage 2 maps `SVDQ_REGION_HIDDEN_Q` plus `SVDQ_REGION_HIDDEN_SCALE` to `SVDQ_REGION_ACCUMULATOR_2`.
- `RunResidualGmmStage` now requires this bridge contract before reaching the execution gate, but still returns
  false. Execution remains fail-closed until the bridge directly reuses or ports the official W4A8 AIC/AIV lifecycle.
- No public `torch_npu.npu_grouped_matmul`, scalar W4A8 GEMM, host-unpacked substitute, or guessed scale formula was
  added.

Regenerated manifest:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Resulting key state:
  `production_fail_closed.residual_gmm_official_bridge_contract_recorded=true`,
  `source_proof.kernel_residual_gmm_official_bridge_contract_recorded=true`,
  `production_fail_closed.residual_gmm_execution_enabled=false`,
  `production_admission.remaining_execution_requirements.residual_w4a8_gmm_execution_enabled=false`, and
  `production_admission.production_enable_allowed=false`.

Validation:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- `git diff --check -- csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q` -> `46 passed, 16 warnings`

## Stage 2.4 Residual W4A8 Official Tiling Bridge - 2026-06-27

Latest code state:

- Added `SVDQResidualW4A8BridgeTiling` to carry an embedded `DispatchFFNCombineW4A8TilingData` payload for the future
  official residual W4A8 GMM bridge.
- Host tiling now fills the official W4A8 shape contract: `M=m`, `K=hiddenSize`, `N=2*intermediateSize`,
  `listLen=expertPerRank`, `isTransposeB=false`, and `isWeightNz=true`.
- The bridge copies the official CoC constants `m0=128`, `k0=256`, `n0=256`, `swizzleDirect=1`,
  `swizzleOffset=7`, `ubMoveNum=16*1024`, `pValue=1`, `commNpuSplit=worldSize`, `commDataSplit=1`, and
  `lenPerLoop=m0*n0/2`.
- The bridge reuses `dispatchRouting.initRoutingQuantTilingKey` and
  `dispatchRouting.moeInitRoutingQuantV2TilingData` so the future residual W4A8 path is tied to the official routing
  quant tiling payload instead of a parallel substitute.
- `officialWorkspaceBytes` is computed with the official W4A8 workspace formula using the SVDQ production dimensions.
- `hostExecutionFailClosed=true` is recorded in the bridge, and production host tiling still returns `GRAPH_FAILED`.
  Residual W4A8 GMM execution remains disabled until the official GMM2 lifecycle gates pass.

Regenerated manifest:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Resulting key state:
  `production_fail_closed.residual_gmm_official_tiling_bridge_recorded=true`,
  `source_proof.kernel_residual_gmm_official_tiling_bridge_recorded=true`,
  `production_fail_closed.residual_gmm_execution_enabled=false`,
  `production_admission.remaining_execution_requirements.residual_w4a8_gmm_execution_enabled=false`, and
  `production_admission.production_enable_allowed=false`.

Validation:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- `git diff --check -- csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q` -> `46 passed, 16 warnings`

## Stage 2.4 Production Hidden Quant AIV Source Boundary - 2026-06-27

Latest code state:

- Added production AIV hidden quantization source helpers in `dispatch_ffn_combine_w4_a8_svdq.h`.
- The non-routing residual quant stage now dispatches `RunResidualHiddenQuantAIV` for
  `SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN`; routed input quantization still uses the existing official
  `moe_init_routing_quant_v2` path.
- `RunResidualHiddenQuantAIV` reads BF16 hidden from `SVDQ_REGION_HIDDEN`, computes one FP32 scale per active routed
  row with vector `Abs/ReduceMax`, writes the scale to `SVDQ_REGION_HIDDEN_SCALE`, quantizes BF16 hidden with vector
  casts/clamps, and packs the result into the official high/low INT4 byte layout in `SVDQ_REGION_HIDDEN_Q`.
- The packing path follows the same high/low layout proven by the isolated mixed-epilogue debug readback:
  high nibbles at `row * k + column / 2`, low nibbles at `row * k + k / 2 + column / 2`.
- No public `torch_npu.npu_grouped_matmul` path, scalar W4A8 GEMM, host-unpacked substitute, or guessed scale formula
  was added.
- Residual W4A8 GMM1/GMM2 production execution remains fail-closed until it is wired through the official W4A8
  AIC/AIV lifecycle. Production host tiling still returns `GRAPH_FAILED`.

Regenerated manifest:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Resulting key state:
  `production_fail_closed.residual_hidden_quant_execution_enabled=true`,
  `source_proof.kernel_residual_hidden_quant_aiv_execution_enabled=true`,
  `source_proof.kernel_residual_hidden_quant_scalar_execution_enabled=false`,
  `production_admission.remaining_execution_requirements.residual_hidden_quant_execution_enabled=true`,
  `production_admission.remaining_execution_requirements.residual_w4a8_gmm_execution_enabled=false`, and
  `production_admission.production_enable_allowed=false`.
- Remaining source execution gaps:
  residual W4A8 GMM1/GMM2 production execution and four-NPU target-model E2E validation.

Validation:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- `git diff --check -- csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q` -> `46 passed, 16 warnings`

## Stage 2.4 Production Mixed AIV Epilogue Source Boundary - 2026-06-27

Latest binding constraint reread:

- Reread `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` before this source patch.
- No official GMM2 lifecycle, V2C/C2V flag protocol, packed W4 layout, W4A8 scale formula, or public
  `torch_npu.npu_grouped_matmul` path was changed or reinterpreted.
- This source patch depends on historical Stage 2.2 Gate A/B/C recheck evidence listed above. Under the current
  appendix it does not claim Stage 2.2 admission, does not unblock Stage 2.3+, and does not enable production host
  tiling.

Latest code state:

- Added production AIV source helpers for the two mixed epilogues in
  `dispatch_ffn_combine_w4_a8_svdq.h`.
- Stage 0, `RunMixedSwiGLUEpilogueAIV`, reads BF16 W4A8 residual gate/up output from
  `SVDQ_REGION_ACCUMULATOR_1` and BF16 low-rank gate/up output from `SVDQ_REGION_PROJECTION_1`, adds them in FP32,
  applies optional SwiGLU clamping and vector `Exp/Div/Mul`, then writes BF16 hidden to `SVDQ_REGION_HIDDEN`.
- Stage 1, `RunMixedOutputEpilogueAIV`, reads BF16 W4A8 residual down output from `SVDQ_REGION_ACCUMULATOR_2` and
  BF16 low-rank down output from `SVDQ_REGION_PROJECTION_2`, adds them in FP32, and writes BF16 peer output to
  `SVDQ_REGION_PEER_OUTPUT`.
- Both helpers return true without vector work on AIC cores and do the vector work only on AIV cores.
- The forbidden scalar helper family remains absent:
  `RunMixedOutputEpilogueStage`, `RunMixedSwiGLUEpilogueStage`, `LoadMixedEpilogueResidualBF16`,
  `LoadMixedEpilogueLowRankBF16`, `StoreMixedEpilogueOutputBF16`, `SiluFloat`, and `ExpApproxFloat`.
- Residual hidden quantization and residual W4A8 GMM1/GMM2 production execution remain fail-closed.
- Production host tiling remains fail-closed with `GRAPH_FAILED`; four-NPU target-model E2E validation remains open.

Regenerated manifest:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Resulting key state:
  `production_fail_closed.mixed_output_epilogue_execution_enabled=true`,
  `production_fail_closed.mixed_swiglu_epilogue_execution_enabled=true`,
  `source_proof.kernel_mixed_output_epilogue_aiv_execution_enabled=true`,
  `source_proof.kernel_mixed_swiglu_epilogue_aiv_execution_enabled=true`,
  `source_proof.kernel_mixed_output_epilogue_scalar_execution_enabled=false`,
  `source_proof.kernel_mixed_swiglu_epilogue_scalar_execution_enabled=false`,
  `production_admission.production_enable_allowed=false`, and
  `production_admission.host_tiling_must_remain_fail_closed=true`.
- Remaining source execution gaps:
  residual hidden quant, residual W4A8 GMM1/GMM2 production execution, and four-NPU target-model E2E validation.

Validation:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `git diff --check -- csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q` -> `46 passed, 16 warnings`
- Manifest regeneration command above passed and wrote
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`.

## Stage 2.4 Production Final Combine Source Boundary - 2026-06-27

Latest code state:

- Integrated the production final-combine source boundary through the official token-unpermute helper used by both
  official BF16 and official W4A8 `CombineV2` paths:
  `KernelMoeTokenUnpermute<bfloat16_t, int32_t, float, true>`.
- Added `SVDQFinalCombineTiling` with `MoeTokenUnpermuteTilingData` and built it using
  `MoeTokenUnpermuteTiling(info.m * info.topK, info.hiddenSize, info.topK, ...)`.
- `RunFinalCombine()` now calls the official helper with the SVDQ peer-output region, expanded-row index region,
  top-k probabilities, and final output pointer.
- The hand-written scalar final-combine substitute remains absent. The manifest now distinguishes
  `kernel_final_combine_official_unpermute_execution_enabled=true` from
  `kernel_final_combine_scalar_execution_enabled=false`.
- This does not change W4A8 GMM2 producer/consumer lifecycle, hidden quantization, mixed AIV epilogues, or production
  host tiling. Host tiling remains fail-closed with `GRAPH_FAILED`.
- The public `torch_npu.npu_grouped_matmul` path was not used or modified.

Regenerated manifest:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Resulting key state:
  `production_fail_closed.dispatch_routing_execution_enabled=true`,
  `production_fail_closed.final_combine_execution_enabled=true`,
  `production_admission.remaining_execution_requirements.final_combine_execution_enabled=true`,
  `production_admission.production_enable_allowed=false`, and
  `production_admission.host_tiling_must_remain_fail_closed=true`.
- Remaining source execution gaps:
  residual hidden quant, residual W4A8 GMM1/GMM2 production execution, mixed SwiGLU epilogue, mixed output epilogue,
  and four-NPU target-model E2E validation.

Validation:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `git diff --check -- csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q` -> `46 passed, 16 warnings`

## Stage 2.4 Production Dispatch Routing Source Boundary - 2026-06-27

Latest code state:

- Read `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` as a binding constraint. This
  is now superseded by the later 2026-06-27 reread above: Stage 2.2 is reopened as `FAIL / IN PROGRESS`, and prior
  pass evidence is historical only.
- Added a production SVDQ BF16 dispatch-routing source boundary that reuses the official W4A8 routing helper set
  through an SVDQ-local route-only wrapper over `InnerMoeInitRoutingV2TilingData`. This prepares routed BF16 rows and
  expanded-row metadata before low-rank and residual stages.
- Added separate route-only routing tiling/workspace fields:
  `bf16RoutingTilingKey`, `bf16RoutingWorkspaceBytes`, and `moeInitRoutingV2TilingData`.
- Kept the existing W4A8 quant-routing tiling/workspace for routed-input dynamic quantization and moved
  `DispatchQuantRoutingTempWorkspace()` after the BF16 route-only workspace so the temporary buffers do not overlap.
- Production host tiling remains fail-closed with `GRAPH_FAILED`; this is not production admission and does not enable
  W4A8 GMM1/GMM2, mixed epilogues, final combine, or four-NPU serving.
- The public `torch_npu.npu_grouped_matmul` path was not used or modified.

Regenerated manifest:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Resulting key state:
  `production_fail_closed.dispatch_routing_execution_enabled=true`,
  `production_admission.remaining_execution_requirements.dispatch_routing_execution_enabled=true`,
  `production_admission.production_enable_allowed=false`, and
  `production_admission.host_tiling_must_remain_fail_closed=true`.
- Remaining source execution gaps:
  residual hidden quant, residual W4A8 GMM1/GMM2 production execution, mixed SwiGLU epilogue, mixed output epilogue,
  final combine, and four-NPU target-model E2E validation.

Validation:

- `python -m py_compile tools/svdq_kernel_contract_manifest.py`
- `git diff --check -- csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq.h csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py`
- `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q` -> `46 passed, 16 warnings`

## Historical: Stage 2.2 Official-Path Gate Reconciled With Evidence - 2026-06-27

Historical section superseded by the 2026-06-27 reread of the GMM2 official-path appendix. The appendix
`svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding for acceptance criteria:
Stage 2.2 cannot pass from source inspection, final-output-only summaries, public grouped matmul experiments, scalar
substitutes, or tolerance relaxation. This historical section previously interpreted the dedicated Stage 2.2
evidence as satisfying those criteria; the current manifest no longer consumes that evidence for admission and
records it only as historical.

Evidence consumed:

- Stage 2.2 dedicated summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_trunc13_reference_true_top1_expert0.json`
- Stage 2.3 top-k 8 final-combine summary:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.json`
- Updated production admission manifest:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

Stage 2.2 accepted fields:

- `official_gmm2_entry_reached=true`
- `official_gmm2_loop_count=8`
- `official_gmm2_active_tile_count=8`
- `official_gmm2_loop_stats_valid=true`
- `official_gmm2_active_tile_count_nonzero=true`
- `official_gmm2_aic_raw_output_finite=true`
- `official_gmm2_aic_raw_output_nonzero=true`
- `official_gmm2_aic_reference_passed=true`
- `official_gmm2_accumulator_int32_reference_passed=true`
- `official_gmm2_c2v_handoff_verified=true`
- `official_gmm2_post_dequant_finite=true`
- `official_gmm2_post_dequant_nonzero=true`
- `official_gmm2_post_dequant_reference_passed=true`
- `official_gmm2_numerical_gate_passed=true`
- Gate A row identity, packed hidden, post-override hidden readback, hidden scale, and padded-row checks pass.
- Public grouped matmul is not used, and production SVDQ host tiling remains fail-closed.

Machine-checkable guardrail update:

- `tools/svdq_kernel_contract_manifest.py` now reads the dedicated Stage 2.2 evidence file and fails closed if the
  file is missing, invalid, or lacks any required appendix status field.
- Stage 2.3 evidence is accepted only when both the Stage 2.2 official-GMM2 gate and Stage 2.3 same-routing/final
  combine gate pass.
- Production enablement still remains `false`; passing isolated gates does not enable host tiling or four-NPU service.

## Historical: Stage 2.2 Official-Path Appendix Supersession - 2026-06-27

Historical section superseded by the evidence-backed manifest parser on 2026-06-27. The new binding appendix
`svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was read and applied as a higher-priority
constraint. It restores Stage 2.2 to `FAIL / IN PROGRESS` and blocks Stage 2.3+, regardless of later historical
composition/final-combine summaries, until the official W4A8 GMM2 lifecycle gate passes with explicit Gate A/B/C
readbacks.

Required current interpretation:

1. Stage 2.0 and Stage 2.1 remain passed.
2. Stage 2.2 is open. The active problem is the official W4A8 GMM2 producer/consumer lifecycle and state
   reconstruction for modified hidden input, specifically the all-zero official post-dequant readback against a
   nonzero unfused official-contract reference.
3. Stage 2.3 same-routing composition, SVDQ down composition, final combine, Stage 2.4 production admission, and
   four-NPU target-model serving are blocked until Stage 2.2 passes.
4. The only acceptable source of truth for W4A8 GMM2 remains the official `dispatch_ffn_combine_w4_a8` AIC/AIV path.
   Public `torch_npu.npu_grouped_matmul`, scalar substitutes, guessed scale formulas, and host-unpacked W4A8 GEMMs
   remain forbidden.
5. Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

Machine-checkable guardrail added:

- `tools/svdq_kernel_contract_manifest.py` now emits `production_admission.stage2_2_official_gmm2_gate` with
  `status=fail_in_progress`, `passed=false`, and required fields for `official_gmm2_aic_raw_output_*`,
  `official_gmm2_c2v_handoff_verified`, `official_gmm2_post_dequant_*`, and
  `official_gmm2_numerical_gate_passed`.
- Existing Stage 2.3 top-k/final-combine evidence is preserved as historical evidence only. The manifest records
  `historical_passed_under_superseded_contract=true` when the old summary satisfies the old checks, but it forces
  `stage2_3_real_checkpoint_composition_gate.passed=false` and
  `production_admission.stage2_3_isolated_gate_passed=false` while Stage 2.2 is open.

No kernel behavior was changed in this update. The next behavioral patch must first identify a specific deviation in
the official-vs-debug state table and tie it to an exact official source counterpart.

## Historical: Stage 2.3 Final-Combine Gate Passed - 2026-06-27T03:24Z

Historical section superseded by the Stage 2.2 official-path appendix on 2026-06-27. The previous final-combine failure was caused by using the
wrong index convention in the probe/oracle: it passed an expert-contiguous input-row to flattened `[token, top_k]`
slot map. The official `KernelMoeTokenUnpermute` path consumes the inverse convention: flattened token-major output
slots point to input rows in the permuted/expert-contiguous peer-output buffer.

Files changed:

- `vllm_ascend/quantization/methods/svdq_post_load.py`
- `tools/svdq_w4a8_tap_mixed_epilogue_probe.py`
- `tools/svdq_kernel_contract_manifest.py`
- `tests/ut/quantization/test_svdq_post_load.py`
- `tests/ut/ops/test_svdq_moe_abi.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Exact behavior added:

- `build_svdq_final_combine_reference(..., expanded_row_idx=...)` now mirrors the official token-unpermute
  chunk-gather convention used by
  `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/unpermute/moe_token_unpermute.h`.
- The explicit `token_indices/topk_indices` reference mode remains available for row-to-token references.
- The real Stage 2.3 tap/mixed probe now constructs final-combine indices as token-major output slots to
  permuted input rows and records
  `index_semantics=official_token_major_output_slots_to_permuted_input_rows`.
- The production-admission manifest now consumes
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.json`
  and requires the official final-combine index semantics in addition to exact-zero final-combine output.

Validation commands:

- Syntax and whitespace:
  `python -m py_compile vllm_ascend/quantization/methods/svdq_post_load.py tools/svdq_w4a8_tap_mixed_epilogue_probe.py`
  `git diff --check -- vllm_ascend/quantization/methods/svdq_post_load.py tools/svdq_w4a8_tap_mixed_epilogue_probe.py tests/ut/quantization/test_svdq_post_load.py`
- Focused static regressions:
  `python -m pytest tests/ut/quantization/test_svdq_post_load.py tests/ut/ops/test_svdq_moe_abi.py -q`
  passed with `56 passed, 16 warnings`.
- A tiny NPU convention probe with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` confirmed that
  `torch_npu.npu_moe_token_unpermute` gathers by token-major chunks of input-row indices.
- Real-device top-k 8 final-combine probe command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_tap_mixed_epilogue_probe.py --require-npu --top-k 8 --route-experts 0 1 2 3 4 5 6 7 --local-num-experts 8 --num-tokens 4 --max-output-size 64 --summary-name stage2/phase_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.json`

Evidence paths:

- Top-k 8 fixed-index final-combine NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_fixedidx_preflight_npus.log`
- Top-k 8 fixed-index final-combine Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_fixedidx_preflight_python.log`
- Top-k 8 fixed-index final-combine active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_fixedidx_preflight_processes.log`
- Top-k 8 fixed-index final-combine probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.log`
- Top-k 8 fixed-index final-combine probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.exitcode`
  contains `0`.
- Top-k 8 fixed-index final-combine probe summary:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.json`
- Updated production admission manifest:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Updated production admission manifest log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_4_production_admission_manifest_fixedidx.log`

Real-device result:

- Overall probe: `passed: true`.
- Shape: `num_tokens=4`, `top_k=8`, `active_rows=32`, `hidden_size=2048`, `intermediate_size=512`.
- Stage flags all true:
  `first_mixed_epilogue`, `official_gmm2_from_svdq_hidden`, `gate_mixed`, `up_mixed`, `hidden_bf16`,
  `hidden_scale`, `hidden_q`, `down_mixed`, `out_bf16`, `final_combine_output`, and `same_routing_identity`.
- Same-routing manifest checks all true, including
  `final_combine_consumes_mixed_down_peer_output=true`,
  `final_combine_output_validated=true`, and
  `same_source_token_payload_across_topk_slots_proven=true`.
- Final-combine error:
  active shape `[4, 2048]`, actual dtype `torch.bfloat16`, expected dtype `torch.bfloat16`,
  finite actual/expected/diff, max abs `0.0`, mean abs `0.0`.
- Historical production admission gate under the superseded contract:
  `status=passed`, `passed=true`, `stage2_3_isolated_gate_passed=true`,
  `exact_numerical_flags.final_combine_output_error_zero=true`,
  `required_final_combine_flags.real_final_combine_uses_official_index_semantics=true`,
  `production_enable_allowed=false`, and `host_tiling_must_remain_fail_closed=true`.

Current interpretation:

1. Stage 2.3 now validates the complete isolated real-checkpoint path through final combine:
   official W4A8 GMM1, actual SVDQ BF16 gate/up, mixed AIV SwiGLU, official W4A8 GMM2 from SVDQ-modified hidden,
   actual SVDQ BF16 down, final mixed output, and official probability-weighted token unpermute.
2. Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed. The next required work is production fused-operator
   integration and then four-NPU target-model validation.
3. The public `torch_npu.npu_grouped_matmul` path remains out of scope for W4A8 GMM semantics.

## Stage 2.3 Final-Combine Gate Blocked - 2026-06-27T03:12Z

This is the latest handoff before the environment rebuild. It extends the Stage 2.3 real-checkpoint top-k 8 probe
with the actual final unpermute/combine boundary that consumes the mixed down peer output. It does not change the
previous W4A8 GMM2 or mixed AIV result: those remain exact-zero against their references. The new final-combine
boundary is finite and close, but it fails the strict exact-zero gate, so it must not be reported as passed.

Files changed:

- `tools/svdq_w4a8_tap_mixed_epilogue_probe.py`
- `tools/svdq_kernel_contract_manifest.py`
- `tests/ut/ops/test_svdq_moe_abi.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Exact behavior added:

- The tap/mixed probe now builds expert-contiguous `expanded_row_idx` for the real top-k 8 routed rows and validates
  the final mixed down peer output through `torch_npu.npu_moe_token_unpermute`.
- The comparison oracle is `build_svdq_final_combine_reference`; the probe records `real_final_combine`,
  `stage_errors.final_combine_output`, and same-routing checks for the final-combine input/output boundary.
- The production-admission manifest now consumes
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_topk8_finalcombine_experts0_7.json`
  and requires `final_combine_output`, `final_combine_consumes_mixed_down_peer_output`,
  `final_combine_output_validated`, and exact-zero `final_combine_output_error_zero`.
- Production admission remains fail-closed and now fails on the final-combine gate instead of passing on the older
  pre-final-combine Stage 2.3 evidence.

Validation commands:

- Syntax and whitespace:
  `python -m py_compile tools/svdq_w4a8_tap_mixed_epilogue_probe.py tools/svdq_kernel_contract_manifest.py`
  `git diff --check -- tools/svdq_w4a8_tap_mixed_epilogue_probe.py tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py`
- Focused static ABI regression:
  `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
  passed with `45 passed, 16 warnings`.
- Real-device preflight used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`; Python preflight reported four visible
  Ascend910B4 logical devices and selected logical NPU 0.
- Top-k 8 final-combine probe command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_tap_mixed_epilogue_probe.py --require-npu --top-k 8 --route-experts 0 1 2 3 4 5 6 7 --local-num-experts 8 --num-tokens 4 --max-output-size 64 --summary-name stage2/phase_stage2_real_composition_topk8_finalcombine_experts0_7.json`
- Production admission manifest regeneration:
  `python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

Evidence paths:

- Top-k 8 final-combine NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_preflight_npus.log`
- Top-k 8 final-combine Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_preflight_python.log`
- Top-k 8 final-combine active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_preflight_processes.log`
- Top-k 8 final-combine probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_experts0_7.log`
- Top-k 8 final-combine probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_finalcombine_experts0_7.exitcode`
  contains `1`.
- Top-k 8 final-combine probe summary:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_topk8_finalcombine_experts0_7.json`
- Updated production admission manifest:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Updated production admission manifest log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_4_production_admission_manifest_finalcombine.log`

Real-device result:

- Overall probe: `passed: false`.
- Shape: `num_tokens=4`, `top_k=8`, `active_rows=32`, `hidden_size=2048`, `intermediate_size=512`.
- Still passing exact-zero stages:
  `first_mixed_epilogue`, `official_gmm2_from_svdq_hidden`, `gate_mixed`, `up_mixed`, `hidden_bf16`,
  `hidden_scale`, `hidden_q`, `down_mixed`, and `out_bf16`.
- Final-combine failing stage:
  `stage_passed.final_combine_output=false`.
- Final-combine error:
  active shape `[4, 2048]`, actual dtype `torch.bfloat16`, expected dtype `torch.bfloat16`,
  finite actual/expected/diff, max abs `7.703783921897411e-07`, mean abs `1.5680409148899344e-07`.
- Same-routing checks still true through the input boundary:
  `expert_token_total_matches_active_rows=true`,
  `same_canonical_hidden_feeds_svdq_down_and_w4a8_hidden_quant=true`,
  `official_gmm2_output_feeds_final_mixed_residual_down=true`,
  `final_mixed_output_is_final_combine_input=true`,
  `final_combine_consumes_mixed_down_peer_output=true`,
  and `same_source_token_payload_across_topk_slots_proven=true`.
- Same-routing final-combine validation is false:
  `final_combine_output_validated=false`, therefore `same_routing_identity=false`.
- Updated production admission gate:
  `status=failed`, `passed=false`, `stage2_3_isolated_gate_passed=false`,
  `exact_numerical_flags.final_combine_output_error_zero=false`,
  `required_final_combine_flags.real_final_combine_passed=false`,
  `production_enable_allowed=false`, and `host_tiling_must_remain_fail_closed=true`.

Current interpretation:

1. The W4A8 GMM2 official lifecycle and both mixed AIV additions remain validated with exact-zero error.
2. The new active blocker is the final weighted token-unpermute validation on real peer-output rows. The failure is
   finite and small, but strict exact-zero was required and was not relaxed.
3. The next session should inspect the `torch_npu.npu_moe_token_unpermute` accumulation/order contract or the final
   combine oracle before any production tiling enablement. Do not reinterpret this failed final-combine evidence as a
   passed numerical gate.
4. The new `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was read. It remains binding
   for future W4A8 GMM2 work: preserve the official `dispatch_ffn_combine_w4_a8` lifecycle, do not use public
   grouped matmul as an oracle, and keep production fail-closed.

## Stage 2.4 Production Admission Gate Added - 2026-06-27T03:05Z

This section is the latest handoff for production integration. It does not supersede the Stage 2.3 numerical pass:
Stage 2.3 remains passed. It adds a machine-checkable admission gate that consumes the Stage 2.3 top-k 8 evidence
and keeps production tiling blocked until the remaining fused execution paths are implemented and validated.

Files changed:

- `tools/svdq_kernel_contract_manifest.py`
- `tests/ut/ops/test_svdq_moe_abi.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Exact behavior added:

- `build_manifest(..., evidence_dir=...)` now records `production_admission`.
- The admission gate reads
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_topk8_experts0_7.json`.
- It verifies the Stage 2.3 summary is the real-checkpoint top-k 8 gate:
  `num_tokens=4`, `top_k=8`, `active_rows=32`, `hidden_size=2048`, `intermediate_size=512`.
- It requires all Stage 2.3 stage flags to pass:
  first mixed epilogue, official GMM2 from SVDQ hidden, gate/up hidden quant, down mixed output, final BF16 output,
  and same-routing identity.
- It requires all same-routing checks to pass, including
  `same_source_token_payload_across_topk_slots_proven=true`.
- It requires all official GMM2-from-SVDQ-hidden checks to pass.
- It requires exact zero error for official GMM2, final mixed down, and final BF16 output.
- It explicitly reports `production_enable_allowed=false` and `host_tiling_must_remain_fail_closed=true`.

Validation commands:

- Syntax and whitespace:
  `python -m py_compile tools/svdq_kernel_contract_manifest.py`
  `git diff --check -- tools/svdq_kernel_contract_manifest.py tests/ut/ops/test_svdq_moe_abi.py`
- Focused static ABI regression:
  `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
  passed with `45 passed, 16 warnings`.
- Manifest generation:
  `python tools/svdq_kernel_contract_manifest.py --evidence-dir /root/workspace/lza/svdq_clean_evidence --output /root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`

Evidence paths:

- Manifest output:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_4_production_admission_manifest.json`
- Manifest log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_4_production_admission_manifest.log`

Validated manifest result:

- `production_admission.stage2_3_real_checkpoint_composition_gate.evidence_found: true`.
- `production_admission.stage2_3_real_checkpoint_composition_gate.status: passed`.
- `production_admission.stage2_3_real_checkpoint_composition_gate.passed: true`.
- `production_admission.stage2_3_isolated_gate_passed: true`.
- `production_admission.production_enable_allowed: false`.
- `production_admission.host_tiling_must_remain_fail_closed: true`.
- Remaining execution requirements are all still false:
  `dispatch_routing_execution_enabled`,
  `residual_hidden_quant_execution_enabled`,
  `residual_w4a8_gmm_execution_enabled`,
  `mixed_swiglu_epilogue_execution_enabled`,
  `mixed_output_epilogue_execution_enabled`,
  `final_combine_execution_enabled`,
  and `four_npu_target_model_e2e_validated`.

Current interpretation:

1. The codebase now has a machine-checkable bridge from the Stage 2.3 real-device evidence to production admission.
2. The admission gate prevents treating the isolated Stage 2.3 pass as production readiness.
3. The next implementation work must replace the production fused-kernel execution stubs with validated official
   production paths, starting with dispatch/routing and official W4A8 residual GMM integration, while preserving the
   official contracts validated in Stage 2.2 and Stage 2.3.

## Stage 2.3 Real-Checkpoint Top-k 8 Composition Pass - 2026-06-27T02:57Z

This section is the latest authoritative handoff. It supersedes the `2026-06-27T02:43Z` synthetic same-routing
section for Stage 2.3 status. Stage 2.0, Stage 2.1, and Stage 2.2 remain passed. Production
`DispatchFFNCombineW4A8SVDQ` remains fail-closed; this pass validates the isolated real-device numerical gate only.

Files changed:

- `tools/svdq_w4a8_tap_mixed_epilogue_probe.py`
- `tests/ut/ops/test_svdq_moe_abi.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Exact behavior added:

- The real-checkpoint tap/mixed probe now performs a full isolated Stage 2.3 sequence:
  1. official W4A8 debug readback supplies the real residual GMM1 gate/up output;
  2. actual SVDQ gate/up BF16 debug output is added through the mixed epilogue debug op;
  3. the first mixed epilogue device output supplies canonical BF16 hidden, hidden INT8, packed INT4 hidden, and
     hidden scale;
  4. official `svdq_w4a8_gmm2_debug_readback` relaunches W4A8 GMM2 from that SVDQ-modified hidden boundary;
  5. actual SVDQ down BF16 debug output consumes the same canonical hidden;
  6. the final mixed epilogue adds official W4A8 residual down and actual SVDQ down output.
- The probe emits `stage2_3_same_routing_manifest`, including hashes for routed input, residual GMM1 rows, SVDQ
  gate/up rows, canonical hidden, hidden quant tensors, official GMM2 residual down, SVDQ down output, final mixed
  down output, and final BF16 output.
- The manifest now proves same source-token payload equality across all top-k slots for the routed rows, not just
  the easier top-1 case.
- The official GMM2 relaunch reuses the accepted Stage 2.2 official lifecycle: `DispatchAndCombine` with only the
  hidden packed INT4 and hidden-scale boundary overlaid. It does not use public `torch_npu.npu_grouped_matmul`.

Validation commands:

- Syntax and whitespace:
  `python -m py_compile tools/svdq_w4a8_tap_mixed_epilogue_probe.py`
  `git diff --check -- tools/svdq_w4a8_tap_mixed_epilogue_probe.py tests/ut/ops/test_svdq_moe_abi.py`
- Focused static ABI regression:
  `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
  passed with `44 passed, 16 warnings`.
- Real-device preflight used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`; Python preflight reported four visible
  Ascend910B4 logical devices and selected logical NPU 0.
- Top-k 8 probe command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_tap_mixed_epilogue_probe.py --require-npu --top-k 8 --route-experts 0 1 2 3 4 5 6 7 --local-num-experts 8 --num-tokens 4 --max-output-size 64 --summary-name stage2/phase_stage2_real_composition_topk8_experts0_7.json`

Evidence paths:

- Top-k 8 NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_preflight_npus.log`
- Top-k 8 Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_preflight_python.log`
- Top-k 8 active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_preflight_processes.log`
- Top-k 8 probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_experts0_7.log`
- Top-k 8 probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_real_composition_topk8_experts0_7.exitcode`
  contains `0`.
- Top-k 8 probe summary:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_topk8_experts0_7.json`
- Additional top-1 sanity run:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_real_composition_true_top1_expert0.json`
  also passed.

Real-device top-k 8 result:

- Overall probe: `passed: true`.
- Shape: `num_tokens=4`, `top_k=8`, `active_rows=32`, `hidden_size=2048`, `intermediate_size=512`.
- Rank metadata: `gate_rank=64`, `up_rank=64`, `down_rank=64`, `gate_rank_offset=0`, `up_rank_offset=64`.
- `stage_passed.first_mixed_epilogue: true`.
- `stage_passed.official_gmm2_from_svdq_hidden: true`.
- `stage_passed.down_mixed: true`.
- `stage_passed.out_bf16: true`.
- `stage_passed.same_routing_identity: true`.
- Same-routing manifest checks:
  `expert_token_total_matches_active_rows=true`,
  `same_canonical_hidden_feeds_svdq_down_and_w4a8_hidden_quant=true`,
  `official_gmm2_output_feeds_final_mixed_residual_down=true`,
  `final_mixed_output_is_final_combine_input=true`,
  `same_source_token_payload_across_topk_slots_proven=true`.
- Official GMM2 from SVDQ hidden checks:
  `official_gmm2_entry_reached=true`,
  `gate_a_input_boundary_passed=true`,
  `official_gmm2_post_dequant_finite=true`,
  `official_gmm2_post_dequant_nonzero=true`,
  `official_gmm2_post_dequant_reference_passed=true`,
  `official_gmm2_numerical_gate_passed=true`.
- Official GMM2 strict reference error: active shape `[32, 2048]`, max abs `0.0`, mean abs `0.0`.
- Final mixed down error: active shape `[32, 2048]`, max abs `0.0`, mean abs `0.0`.
- Final BF16 output error: active shape `[32, 2048]`, max abs `0.0`, mean abs `0.0`.

Current interpretation:

1. Stage 2.3 isolated real-checkpoint same-routing composition is now passed for a single-device top-k 8 route
   through experts 0-7.
2. The pass validates the AIV mixed additions the user asked about: W4A8 GMM1 plus BF16 gate/up, and official W4A8
   GMM2 plus BF16 down, both on real checkpoint factors and official residual W4A8 outputs.
3. This does not enable production `DispatchFFNCombineW4A8SVDQ`; host tiling remains fail-closed until the validated
   components are integrated into the production fused operator and four-NPU end-to-end gates pass.

## Stage 2.3 Official GMM2 Path Constraint Refresh - 2026-06-27

The new handoff requirement file was read:

- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`

Binding interpretation for the next environment rebuild:

- The appendix remains binding for any future W4A8 GMM2 lifecycle work: do not change V2C/C2V flags, producer or
  consumer ownership, token state, workspace offsets, loop state, barriers, D2 source regions, `BlockEpilogue2`,
  `CombineV2`, or final drains unless the change is tied to an exact official
  `dispatch_ffn_combine_w4_a8` source location.
- The appendix's embedded status text describes the older all-zero Stage 2.2 failure. Later committed evidence in
  this report supersedes that status: commit `97afc57d` passed Stage 2.2 with the official-lifecycle top-1
  expert-0 probe, Gate A, Gate B, C2V, and strict Gate C all passing with max/mean abs `0.0`.
- The current active gate remains Stage 2.3. The only acceptable next progress is a real-checkpoint same-routing
  composition probe that reuses the passed Stage 2.2 official GMM2 output, actual SVDQ down output from the same
  canonical hidden, and a single routed-row identity manifest for both branches.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed. The public `torch_npu.npu_grouped_matmul` path remains
  out of scope and must not be used as a W4A8 semantic oracle.

No source behavior was changed for this refresh. It records the new constraint so a rebuilt environment does not
mistake source inspection, stale all-zero diagnostics, scalar substitutes, or public grouped-matmul experiments for
Stage 2.3 gate progress.

## Stage 2.3 Synthetic Same-Routing Manifest - 2026-06-27T02:43Z

This section is the latest authoritative handoff. It supersedes the `2026-06-27T02:35Z` Stage 2.2 handoff only
for the next active gate status: Stage 2.2 remains passed, and Stage 2.3 now has synthetic Qwen3.5-dimension
same-routing evidence. It does not complete Stage 2.3 because the required real-checkpoint residual W4A8 plus SVDQ
down/final-combine composition is still missing.

Files changed:

- `tools/svdq_composed_pipeline_device_probe.py`
- `tests/ut/ops/test_svdq_moe_abi.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Exact behavior added:

- `svdq_composed_pipeline_device_probe.py` now emits `routing_identity_manifest`.
- The manifest records flattened `[token, top_k]` row identity:
  `routed_row = source_token_id * top_k + top_k_slot`.
- The manifest records per-row identity fields for the first 64 rows: source token, top-k slot, global/local expert
  ID, expert prefix start, expert-local row offset, TP rank, EP rank, and active status.
- It records SHA256 hashes for the shared routed input, canonical hidden, hidden quant branch, SVDQ down hidden
  input, mixed down peer output, final-combine input, and `expanded_row_idx`.
- The probe now includes `routing_identity` in `stage_passed`; a route mismatch fails the probe.

Stage 2.3 checks covered by this synthetic probe:

- First-stage W4A8 placeholder and SVDQ gate/up branch share the same `routed_x` bytes.
- SVDQ down and W4A8 hidden quant consume the same canonical BF16 hidden bytes.
- Final combine consumes the mixed down peer output bytes.
- `expanded_row_idx` is exactly `arange(num_tokens * top_k)`.
- All synthetic routed rows are active with no padding.

Validation commands:

- Syntax and whitespace:
  `python -m py_compile tools/svdq_composed_pipeline_device_probe.py`
  `git diff --check -- tools/svdq_composed_pipeline_device_probe.py tests/ut/ops/test_svdq_moe_abi.py`
- Focused static ABI regression:
  `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
  passed with `44 passed, 16 warnings`.
- Real-device preflight used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`; Python preflight reported four visible
  Ascend910B4 logical devices and selected logical NPU 0.
- Probe command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python tools/svdq_composed_pipeline_device_probe.py --require-npu --summary-name stage2/phase_stage2_composed_routing_manifest_summary.json`

Evidence paths:

- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_composed_routing_manifest_preflight_npus.log`
- Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_composed_routing_manifest_preflight_python.log`
- Active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_composed_routing_manifest_preflight_processes.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_composed_routing_manifest.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_composed_routing_manifest.exitcode`
  contains `0`.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/stage2/phase_stage2_composed_routing_manifest_summary.json`

Real-device result:

- Overall probe: `passed: true`.
- `stage_passed.routing_identity: true`.
- `routing_identity_manifest.checks.first_stage_w4a8_and_svdq_share_routed_x: true`.
- `routing_identity_manifest.checks.svdq_down_and_w4a8_hidden_quant_share_canonical_hidden: true`.
- `routing_identity_manifest.checks.final_combine_uses_mixed_down_peer_output: true`.
- `routing_identity_manifest.checks.expanded_row_idx_matches_arange: true`.
- `routing_identity_manifest.checks.all_rows_active_no_padding: true`.
- Topology: TP size `1`, EP size `1`, `num_tokens=4`, `top_k=8`, routed rows `32`.

Numerical boundaries from the same run:

- `gate_up_rank`: max abs `0.0`, mean abs `0.0`.
- `gate_lowrank`: max abs `0.001953125`, mean abs `9.491109813097864e-05`.
- `up_lowrank`: max abs `0.0009765625`, mean abs `9.620988566894084e-05`.
- `hidden_scale`: max abs `7.275957614183426e-12`, mean abs `6.821210263296962e-13`.
- `down_rank`: max abs `0.0`, mean abs `0.0`.
- `down_lowrank`: max abs `1.52587890625e-05`, mean abs `8.223607892432483e-07`.
- `peer_output`: max abs `0.0`, mean abs `0.0`.
- `final_output`: max abs `0.0`, mean abs `0.0`.

Current interpretation:

1. The composed NPU probe now proves same-routing identity for a synthetic Qwen3.5-dimension full stage order:
   first-stage routed input, canonical hidden, W4A8 hidden quant input, SVDQ down input, mixed down peer output, and
   final combine all use the intended row identity.
2. This is not a Stage 2.3 acceptance pass yet, because it does not use the real-checkpoint W4A8 residual GMM1/GMM2
   outputs, real SVDQ factors, and final real-checkpoint residual-plus-low-rank down path in one route.
3. The next required step is a real-checkpoint Stage 2.3 probe that reuses the Stage 2.2 official GMM2 output,
   actual SVDQ down output from the same canonical hidden, and one routing manifest for both branches.
4. Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

## Stage 2.2 Trunc13 Fixpipe Reference Pass - 2026-06-27T02:35Z

Historical section for the current gate. Stage 2.2 remains passed; Stage 2.3 synthetic same-routing evidence is now
recorded in the `2026-06-27T02:43Z` section above. This section previously superseded the `2026-06-27T02:24Z`
FP16-scale / ULP diagnostic by promoting the source-backed, real-device-proven Fixpipe scale contract into the
strict Stage 2.2 GMM2 unfused reference.

Baseline note:

- The Stage 2 prompt names `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_report_clarified_completed.md`
  as the preferred handoff baseline, but that file is not present in the rebuilt environment. A `find` under
  `/root/workspace/lza` found no clarified report. Current baseline therefore uses this Stage 2 report, the Stage 2
  prompt, the main implementation prompt, and the current appendix constraints.

Files changed:

- `tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py`
- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- `tools/svdq_kernel_contract_manifest.py`
- `tests/ut/ops/test_svdq_moe_abi.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Exact correction:

- Added an optional `zero_low_bits` argument to `_int64_float_bits_to_fp32`.
- Updated only GMM2 unfused and raw-C2 references to reinterpret the low 32 bits of the postloaded W4A8 scale after
  zeroing the low 13 bits.
- Left the GMM1 historical reference path unchanged to avoid reopening accepted GMM1 gates.
- Updated local Stage 2.2 raw-D2 and layout diagnostics to use the same `low32_trunc13` scale contract.
- Updated stale static low-rank ABI checks and manifest source proof to require the official BF16 `BlockMmad`
  execution path, and to reject instantiating the older `SVDQLowRankBF16RankBlockMmad` path. This matches the
  Appendix 5 official BF16 lifecycle direction and does not change kernel behavior.
- Kept all device kernels, packed W4 layout, V2C/C2V flags, workspace offsets, tolerances, public grouped-matmul
  avoidance, and production host tiling unchanged.

Source-backed root cause:

- CANN Fixpipe sources describe dequant factor bits `[31:13]` as the FP32 dequant value:
  - `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/impl/basic_api/dav_m300/kernel_operator_fixpipe_impl.h:226-229`
  - `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/impl/basic_api/dav_m300/kernel_operator_fixpipe_v2_impl.h:406-408`
  - `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/impl/basic_api/dav_c310/kernel_operator_fixpipe_impl.h:521-523`
- The prior diagnostic run proved this was the missing Stage 2.2 contract detail:
  `scale_low32_trunc13_fp16_nearest_even` matched both high and low actual D2 readbacks with max abs `0.0`, mean
  abs `0.0`, and best-key failed count `0`.

Validation commands:

- Syntax and whitespace:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py`
  `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py`
- Focused static ABI regression:
  `python -m pytest tests/ut/ops/test_svdq_moe_abi.py -q`
  passed with `44 passed, 16 warnings`.
- A first run of the same static ABI test failed on stale low-rank source checks that still required
  `SVDQLowRankBF16RankBlockMmad blockMmad(resource);`. The source uses the accepted official BF16
  `SVDQOfficialBF16BlockMmad blockMmad(resource);`; the tests and manifest proof were corrected to encode that
  official lifecycle expectation.
- Real-device preflight used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`; Python preflight reported four visible
  Ascend910B4 logical devices and selected logical NPU 0.
- Probe command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_trunc13_reference_true_top1_expert0.json`

Evidence paths:

- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_trunc13_reference_preflight_npus.log`
- Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_trunc13_reference_preflight_python.log`
- Active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_trunc13_reference_preflight_processes.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_trunc13_reference_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_trunc13_reference_true_top1_expert0.exitcode`
  contains `0`.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_trunc13_reference_true_top1_expert0.json`

Validated status fields from `stage.checks`:

- `gate_a_input_boundary_passed: true`
- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_c2v_handoff_verified: true`
- `official_gmm2_post_dequant_finite: true`
- `official_gmm2_post_dequant_nonzero: true`
- `official_gmm2_post_dequant_reference_passed: true`
- `official_gmm2_numerical_gate_passed: true`
- `gmm2_reference_passed: true`
- `official_gmm2_loop_stats_valid: true`
- `official_gmm2_loop_count: 8`
- `official_gmm2_active_tile_count: 8`
- `official_gmm2_active_tile_count_nonzero: true`

Numerical result:

- Gate A: passes; hidden packed and hidden scale post-override readbacks are exact.
- Gate B: passes; int32 accumulator is finite, nonzero, and exact against the integer reference.
- Gate C: passes; post-dequant active shape `[16, 2048]`, finite/nonzero, NaN count `0`, Inf count `0`.
- Strict unfused GMM2 reference error: max abs `0.0`, mean abs `0.0`, numel `32768`.
- Contract formula recorded in the summary:
  `fp16_nearest_even(high_acc * low32_trunc13(postloaded_weight_scale)) * 16 + fp16_nearest_even(low_acc * low32_trunc13(postloaded_weight_scale)) + scale_bias, then * hidden_x_scale`.

Current interpretation:

1. The previous Gate C failure was a host-reference mismatch at the official Fixpipe/D2 scale contract, not a device
   GMM2, hidden packing, AIC/AIV lifecycle, C2V handoff, or BF16 producer failure.
2. Stage 2.2 is now passed for the isolated single-device real-checkpoint top-1 expert-0 official-lifecycle probe.
3. This pass does not complete Stage 2 overall: Stage 2.3 same-routing two-stage composition, final mixed down,
   production fused-operator integration, and four-NPU E2E validation remain incomplete.
4. Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

## Stage 2.2 FP16-Scale / ULP Diagnostic and Official-Path Table - 2026-06-27T02:24Z

Historical section. Superseded by the `2026-06-27T02:35Z` trunc13 Fixpipe reference pass.
It previously superseded the `2026-06-27T02:13Z` loop-stats section by adding the new mandatory official-path
appendix requirements, an updated official-vs-debug state table for the
current normal official-lifecycle probe, and a fresh real-device run from the exact probe code now in the tree.
No kernel behavior, V2C/C2V flag, workspace offset, packed W4 layout, production host tiling, tolerance, or public
`torch_npu.npu_grouped_matmul` path was changed.

New binding appendix read:

- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Exact probe change:

- Added diagnostic-only `low32_value_fp16` scale candidates to the post-dequant and actual-D2 comparator variant
  maps.
- Added FP16 ULP-distance metrics to the actual-D2-from-accumulator variants.
- The change only expands evidence for the unresolved Fixpipe/D2 boundary. It is not a behavioral patch and does
  not count as a substitute reference or Stage 2.2 pass.

Official source locations inspected for this handoff:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:280-344`:
  W4A8 types, D2 layout, `BlockEpilogue2`, and kernel parameter wiring.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:250-275`:
  normal AIC/AIV roles use `GMM2(params)` and `DispatchAndCombine(params)` when `gmm2OnlyFromPacked=false`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:287-318`:
  official bindings for `cumsumMM`, `gmA2I4_I8`, `gmPerTokenScale2`, `tokenPerExpert`, and `preSumBeforeRank`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:687-793`:
  official `GMM2` loop state, `SYNCFLAGV2C` wait, packed W2/scale access, MMAD call, accumulator debug tap, and
  final AIC drain.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1323-1412`:
  official `DispatchAndCombine` constructs `BlockEpilogue2`, runs the official producer, overlays only external
  GMM2 hidden/scale when requested, emits `SYNCFLAGV2C`, then calls `CombineV2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1448-1520`:
  `CombineV2` waits `SYNCFLAGC2V` and invokes `BlockEpilogue2` on official D2 state.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1554-1588`:
  official workspace offsets through `ptrC2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:114-142`:
  `BlockEpilogue2` token layout, event initialization, and finalization.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:147-266`:
  D2 high/low reads, raw debug taps, high*16+low, aux addition, hidden-scale multiply, FP32 debug tap, and final
  cast.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:277-307`:
  final D routing through `tokenPerExpert`, `preSumBeforeRank`, and peer output offset.

Official-vs-current-debug state table:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | `BlockEpilogue1` inside `DispatchAndCombine`; external override is allowed only after the official producer point. | `GMM2` A matrix through `params.layoutA2`. | `initBuffer` binds `workspaceInfo.ptrA2Int4`; `DispatchAndCombine` overlays per `dequantSum`. | `workspaceInfo.ptrA2Int4`, after first packed hidden region; see kernel lines 293-301 and workspace order lines 1554-1588. | INT4 packed stride `problemShape.n()/2`, D1/A2 row-major offsets. | `SYNCFLAGV2C`. | AIV emits after producer/override work for each sync slice. | `GMM2` waits before group tiles. | Normal path continues through `CombineV2`, then token reset/status cleanup. | Normal Stage 2.2 probe now uses official `DispatchAndCombine` and overlays only validated external packed hidden. Historical GMM2-only helper remains but is not acceptance evidence. |
| hidden scale `gmPerTokenScale2` | `BlockEpilogue1`; external override at the same official point. | `BlockEpilogue2` per-row multiply. | `initBuffer` binds `workspaceInfo.ptrPerTokenScale2`. | `ptrPerTokenScale2`, after `ptrPerTokenScale + maxOutputSize * sizeof(float)`. | One float per active routed row. | `SYNCFLAGV2C` with packed hidden; UB events inside epilogue. | AIV emits after scale is available. | GMM2 waits V2C; `BlockEpilogue2` consumes after C2V. | `BlockEpilogue2::Finalize`. | Readback is exact against Stage 2.1 scale; finite/nonzero. |
| `tokenPerExpert` | Official routing/MC2 state. | `GMM2`, `CombineV2`, `BlockEpilogue2`. | `initBuffer` binds `shmem()+peermemInfo.offsetPeerTokenPerExpert`. | Peer shared memory, `tokenPerExpertLayout = Layout3D(AlignUp(EP * expertPerRank + 1, 128), expertPerRank)`. | Layout3D by dst EP/rank/expert. | Official shared-memory and C2V/V2C lifecycle. | Produced before GMM2/epilogue. | GMM2 and `BlockEpilogue2` read during group processing. | `ResetTokenPerExpert` after combine. | Normal probe preserves official lifecycle; row identity/padded-row evidence still needs fuller report fields beyond top-1 EP=1. |
| `cumsumMM` | `GetCumsumForMMAIV` / official routing state. | `GMM2` and `CombineV2` group loops. | `initBuffer` binds `workspaceInfo.ptrcumsumMM`. | After expanded row index workspace. | `EP * EP * expertPerRank` int32 region. | Official V2C/C2V group protocol. | Before GMM2 group loops. | GMM2 and CombineV2 read at group start. | No separate drain. | Loop-stats relaunch validates total loops=8 and active tiles=8 for top-1 expert 0. |
| `preSumBeforeRank` | Official cross-rank prefix path. | `BlockEpilogue2` final D placement. | `initBuffer` binds `workspaceInfo.ptrSumBeforeRank`. | Later workspace prefix region; after debug GMM regions. | `EP * expertPerRank` int32. | Official routing state. | Before final D routing. | `BlockEpilogue2` reads per destination EP. | Normal cleanup after combine. | Normal probe preserves official state; EP=1 top-1 case does not yet prove multi-EP prefixes. |
| GMM2 AIC input tile state | Official `GMM2`. | `BlockMmad`. | Per group after V2C wait. | `gmA2I4_I8`, W2 `ptrB2`, scale `ptrScale2`, output `gmC2`. | `L1TileShape` scheduler; M is doubled for INT4 high/low rows. | `SYNCFLAGV2C`. | AIV emits after official producer/override. | AIC waits before group tiles. | `blockMmad.SynchronizeBlock()` and `Finalize`. | Official `GMM2` is used; Gate B int32 accumulator exact-match passes. |
| GMM2 accumulator / D2 region | `BlockMmad` and Fixpipe. | `BlockEpilogue2`. | Inside `GMM2` tile loop. | `gmC2`; debug accumulator via `ptrDebugGMM2Accumulator`; FP32 tap via `ptrDebugGMM2`. | D2 high/low halves separated by `n2`; even accumulator rows feed high, odd rows feed low. | `SYNCFLAGC2V`. | AIC finalizes after tile groups. | `CombineV2` waits before epilogue. | CombineV2 drains remaining flags and finalizes epilogue. | Raw int32 Gate B is finite, nonzero, exact-match; actual D2 mismatch is within 2 FP16 ULP for tested variants but Gate C still fails. |
| C2V handoff state | `BlockMmad::Finalize(..., SYNCFLAGC2V)`. | `CombineV2`. | GMM2 group completion. | Cross-core flag state. | Per sync group. | `SYNCFLAGC2V`. | After AIC GMM2 group tiles finish. | `CombineV2` waits before `BlockEpilogue2`. | CombineV2 drains through `BlockEpilogue2::Finalize`. | `official_gmm2_c2v_handoff_verified=true` because post-dequant is finite/nonzero after C2V. |
| `BlockEpilogue2` input state | Official D2 high/low, `gmPerTokenScale2`, MAux2, token state. | `BlockEpilogue2`. | Constructed in `DispatchAndCombine` before `CombineV2`. | `gmC2`, `gmCGMM2`, scale GM, MAux2, shmem `offsetD`. | 32-row AIV split; high/low D2 row offset separated by `n2`. | UB events plus C2V waits. | After C2V wait. | Event waits inside epilogue. | `BlockEpilogue2::Finalize`. | Actual high/low D2 readbacks are captured through raw debug modes and compared to accumulator-derived variants. |
| FP32 post-dequant debug tap | `BlockEpilogue2` W4A8_DEBUG path. | Host probe summary. | After high*16+low, aux, hidden-scale multiply. | `ptrDebugGMM2` / workspace debug GMM2 area. | `maxOutputSize * n2` float. | `EVENT_ID7`. | During epilogue debug copy. | Host observes after op completion. | Epilogue finalization. | Finite/nonzero but strict reference fails: max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`. |

Validation commands:

- Syntax and whitespace:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- Real-device preflight used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` and selected logical NPU 0.
- Probe command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_fp16_scale_ulp_schemafix_true_top1_expert0.json`

Evidence paths:

- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_fp16_scale_ulp_schemafix_preflight_npus.log`
- Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_fp16_scale_ulp_schemafix_preflight_python.log`
- Active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_fp16_scale_ulp_schemafix_preflight_processes.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_fp16_scale_ulp_schemafix_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_fp16_scale_ulp_schemafix_true_top1_expert0.exitcode`
  contains `1`, expected because Stage 2.2 still fails strict Gate C.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_fp16_scale_ulp_schemafix_true_top1_expert0.json`

Validated status fields from `stage.checks`:

- `gate_a_input_boundary_passed: true`
- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_c2v_handoff_verified: true`
- `official_gmm2_post_dequant_finite: true`
- `official_gmm2_post_dequant_nonzero: true`
- `official_gmm2_post_dequant_reference_passed: false`
- `official_gmm2_numerical_gate_passed: false`
- `official_gmm2_loop_stats_valid: true`
- `official_gmm2_loop_count: 8`
- `official_gmm2_active_tile_count: 8`
- `official_gmm2_active_tile_count_nonzero: true`

Numerical result:

- Gate A: passes; hidden packed and hidden scale post-override readbacks are exact.
- Gate B: passes; int32 accumulator actual shape `[32, 2048]`, nonzero, exact mismatch count `0`.
- Gate C: fails strict post-dequant reference tolerance but is finite/nonzero; active shape `[16, 2048]`,
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`, NaN count `0`, Inf count `0`.
- Best post-dequant diagnostic variant remains `scale_low32_fp16_toward_zero` with key
  `[0.00023069977760314941, 1.006269667414017e-05, 20]`.
- FP16-scale narrowing is rejected:
  `scale_low32_value_fp16_fp16_toward_zero` is worse at max abs `0.00037679076194763184`,
  mean abs `1.8746879504760727e-05`; `scale_low32_value_fp16_fp16_nearest_even` is worse at max abs
  `0.0004030466079711914`, mean abs `1.732285090838559e-05`.
- Actual-D2 high half best remains `scale_low32_fp16_toward_zero`:
  max abs `0.00048828125`, mean abs `2.2813444957137108e-05`, best-key failed count `1136`,
  FP16 ULP max `2`, FP16 ULP mean `0.22271728515625`, ULP greater-than-one count `6`.
- Actual-D2 low half best remains `scale_low32_fp16_toward_zero`:
  max abs `0.001953125`, mean abs `7.748615462332964e-05`, best-key failed count `5221`,
  FP16 ULP max `2`, FP16 ULP mean `0.244598388671875`, ULP greater-than-one count `7`.

Confirmed interpretation:

1. The normal Stage 2.2 probe is on the official `DispatchAndCombine` lifecycle with external hidden/scale overlay,
   not the historical synthetic GMM2-only lifecycle.
2. The latest failure is not all-zero output: Gate C output is finite and nonzero, but it still fails strict
   numerical tolerance.
3. FP16-narrowing the low32 scale value does not explain the mismatch.
4. The remaining unresolved boundary is the source-backed Fixpipe/D2 value contract between exact int32
   accumulation and strict `BlockEpilogue2` post-dequant agreement.
5. Stage 2.2 remains `FAIL / IN PROGRESS`; Stage 2.3+, SVDQ down composition, final combine, and production host
   tiling remain blocked.

## Stage 2.2 Official-Lifecycle Loop Stats Diagnostic - 2026-06-27T02:13Z

Historical section. Superseded by the `2026-06-27T02:24Z` FP16-scale / ULP diagnostic and official-path table.
It previously superseded the `2026-06-27` official-lifecycle constraint handoff by adding a source-backed
loop/tile diagnostic to the normal Stage 2.2 probe and validating it on real
Ascend 910B4 hardware. The change does not alter kernel behavior, V2C/C2V flags, workspace offsets, packed W4
format, scale formulas, tolerances, production host tiling, or the public grouped-matmul path.

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`

Exact behavior implemented:

- The probe now records `official_lifecycle_debug_contract`, explicitly documenting that
  `SVDQW4A8GMM2DebugReadback` enters the official `DispatchAndCombine` lifecycle because
  `InitGMM2OnlyFromPacked` sets `gmm2OnlyFromPacked_ = false`.
- For normal post-dequant runs, the probe relaunches the same official-lifecycle debug op with
  `swigluLimit=435000.0`, the existing source-backed loop-stats mode, and records
  `gmm2.loop_stats_from_official_lifecycle`.
- The normal summary now fills `official_gmm2_loop_count`, `official_gmm2_active_tile_count`,
  `official_gmm2_loop_stats_valid`, and `official_gmm2_active_tile_count_nonzero` instead of leaving the loop/tile
  fields as `None`.
- The loop-stats relaunch is diagnostic-only and does not change pass/fail. Stage 2.2 still requires Gate C to pass
  strict reference tolerance.

Official source locations tied to this diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:220-233`:
  `InitGMM2OnlyFromPacked` stores the external hidden/scale pointers and sets `gmm2OnlyFromPacked_ = false`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:250-275`:
  AIC runs `GMM2(params)` and AIV runs `DispatchAndCombine(params)` when `gmm2OnlyFromPacked` is false.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:797-820`:
  debug `swigluLimit` ranges select loop stats or raw D2 readback without introducing a new lifecycle.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:831-920`:
  `WriteGMM2OnlyLoopStats` records active rows, doubled rows, core loops, token state, and cumsum state.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1364-1382`:
  the official AIV path overlays only external GMM2 hidden/scale and then emits the official V2C signal.

Validation commands:

- Syntax check:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- Diff whitespace check:
  `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- NPU/process preflight logs captured with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Real-device normal Gate C probe:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_official_lifecycle_loopstats_true_top1_expert0.json`

Evidence paths:

- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_official_lifecycle_loopstats_preflight_npus.log`
- Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_official_lifecycle_loopstats_preflight_python.log`
- Active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_official_lifecycle_loopstats_preflight_processes.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_official_lifecycle_loopstats_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_official_lifecycle_loopstats_true_top1_expert0.exitcode`
  contains `1`, expected because Stage 2.2 still fails strict Gate C.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_official_lifecycle_loopstats_true_top1_expert0.json`

Validated status fields from the new summary:

- `official_lifecycle_debug_contract.mode: official DispatchAndCombine lifecycle with external hidden/scale overlay`
- `official_lifecycle_debug_contract.gmm2_only_from_packed: false`
- `official_gmm2_loop_stats_valid: true`
- `official_gmm2_loop_count: 8`
- `official_gmm2_active_tile_count: 8`
- `official_gmm2_active_tile_count_nonzero: true`
- `gate_a_input_boundary_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_post_dequant_reference_passed: false`
- `official_gmm2_numerical_gate_passed: false`

Numerical result:

- Gate A passes and the Stage 2.1 packed hidden/scale override readback remains exact.
- Official-lifecycle loop stats are valid: total active rows `16`, groups with work `1`, total core loops `8`.
- Pre-Fixpipe int32 accumulator Gate B still passes the exact reference.
- Gate C remains finite/nonzero but fails strict post-dequant tolerance:
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.

Current interpretation:

1. The current debug op is no longer just a standalone GMM2-only state machine for the normal Stage 2.2 probe; it
   enters the official `DispatchAndCombine` lifecycle and overlays only the validated external hidden INT4 and
   hidden scale at the official AIV boundary.
2. The old report language describing the current normal probe as synthetic-lifecycle-only is now superseded for
   the normal run, though the historical GMM2-only helper remains in source for loop/debug modes.
3. The unresolved boundary is still after exact int32 accumulation and before strict Gate C post-dequant agreement:
   either the exact official Fixpipe/D2 value contract or a remaining source-backed reference mismatch around D2
   reconstruction must be resolved without changing public grouped matmul, packed W4, or tolerances.
4. Stage 2.2 remains `FAIL / IN PROGRESS`; Stage 2.3 and production remain blocked.

## Stage 2.2 Official-Lifecycle Constraint Handoff - 2026-06-27

Historical section. It superseded the `2026-06-27T01:59Z` direct
accumulator-to-D2 diagnostic as the controlling work constraint. The prior diagnostics remain valid failed
experiments, but they do not count as Stage 2.2 acceptance evidence until the debug operator is proven equivalent to
the successful official `dispatch_ffn_combine_w4_a8` producer/consumer lifecycle.

New binding appendix read:

- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`

Mandatory interpretation:

- Do not modify, reinterpret, or debug public `torch_npu.npu_grouped_matmul`.
- Do not guess scale formulas, repack W4, unpack weights on the host, or substitute scalar GEMM.
- Do not change V2C/C2V flags, ownership, token state, workspace offsets, loop state, or barriers unless the
  change is tied to a documented official source location.
- Do not proceed to Stage 2.3, SVDQ down composition, final combine, or production host tiling while Stage 2.2 is
  open.
- Keep `DispatchFFNCombineW4A8SVDQ` fail-closed.

Official source locations inspected for the lifecycle table:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:250-275`:
  AIC dispatch calls `GMM2OnlyFromPacked` only for the debug-only path, while the normal AIV path enters
  `DispatchAndCombine`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:287-318`:
  workspace-backed official GM regions are bound for `cumsumMM`, packed hidden `gmA2I4_I8`,
  `gmPerTokenScale2`, `tokenPerExpert`, and `preSumBeforeRank`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:687-747`:
  official `GMM2` derives per-expert `currentM` from `cumsumMM`, doubles M for INT4, binds W2/scale tensors by
  expert, then waits `SYNCFLAGV2C` before processing group tiles.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:668`:
  official GMM producer drains via `blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V)`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:944-987`:
  current GMM2-only debug path manually seeds token state and calls `GMM2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:990-1016`:
  current GMM2-only AIV path manually seeds packed hidden/scale/token state, emits synthetic ready signals, then
  calls `CombineV2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1323-1412`:
  successful official AIV path constructs `BlockEpilogue2`, runs `BlockEpilogue1`, optionally overlays external
  GMM2 hidden/scale for debug, emits `SYNCFLAGV2C`, then calls `CombineV2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1448-1525`:
  `CombineV2` initializes `BlockEpilogue2`, waits C2V group flags, calls `BlockEpilogue2`, drains remaining flags,
  and finalizes the epilogue.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:147-262`:
  `BlockEpilogue2` reads high/low D2 halves, applies high*16+low, adds W4A8 auxiliary, multiplies by hidden scale,
  and writes the W4A8 debug FP32 tap.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:277-307`:
  final D writeback uses `tokenPerExpert`, `preSumBeforeRank`, shared-memory offsetD, and official output layout.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1554-1630`:
  official workspace offset order for expanded row indices, `cumsumMM`, scale buffers, packed INT4 hidden buffers,
  GMM debug taps, and `preSumBeforeRank`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp:253-264`:
  host workspace sizing matches the kernel workspace regions.
- `csrc/third_party/catlass/include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp:1945-1970` and
  `csrc/third_party/catlass/include/catlass/gemm/tile/atlasa2/copy_l1_to_fp.hpp:33-56`:
  vector scale movement into Fixpipe is a raw contiguous vector copy; no hidden source-backed scale-layout
  transform has been found.

Official-vs-debug state table, current snapshot:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | `BlockEpilogue1` in `DispatchAndCombine`; optional external override after official producer. | `GMM2` as A matrix through `params.layoutA2`. | Workspace binding in `initBuffer`; rows produced per `dequantSum`. | `workspaceInfo.ptrA2Int4`; after `ptrA1Int4 + maxOutputSize * K`. | `layoutD1` / `layoutA2`, INT4 packed as `problemShape.n()/2`. | V2C group readiness. | Official AIV sets V2C after each sync slot. | GMM2 waits V2C before groups. | Normal path resets token state after combine. | GMM2-only copies `ptrExternalHiddenX` to `gmA2I4_I8` directly; lifecycle-equivalence not proven. |
| hidden scale `gmPerTokenScale2` | `BlockEpilogue1` per-token hidden quant scale; optional external override. | `BlockEpilogue2` post-dequant multiply. | Workspace binding in `initBuffer`. | `workspaceInfo.ptrPerTokenScale2`; after `ptrPerTokenScale + maxOutputSize * sizeof(float)`. | One scalar per routed row. | V2C readiness with packed hidden. | Official AIV sets after scale is available. | GMM2 waits before using rows; `BlockEpilogue2` reads during C2V-drained combine. | Epilogue finalizes UB events. | GMM2-only copies `ptrExternalHiddenScale` directly; row identity/padded-row proof still incomplete. |
| `tokenPerExpert` | Official routing/MC2 token exchange. | `GMM2`, `CombineV2`, `BlockEpilogue2`. | `tokenPerExpert.SetGlobalBuffer(shmem()+offsetPeerTokenPerExpert)`. | Peer shared memory `offsetPeerTokenPerExpert = SegmentSize - 2 * MB_SIZE`. | `Layout3D(AlignUp(EP * expertPerRank + 1, 128), expertPerRank)`. | C2V/V2C plus shared-memory lifecycle. | Produced before GMM2 and combine. | Read by GMM2 and epilogue. | `ResetTokenPerExpert` after combine. | GMM2-only seeds local entries from `externalExpertTokenNums`; official multi-rank layout equivalence not complete. |
| `cumsumMM` | `GetCumsumForMMAIV` / official routing state. | `GMM2` and `CombineV2` group loops. | Workspace binding in `initBuffer`. | `workspaceInfo.ptrcumsumMM`; after expanded row index workspace. | `EP * EP * expertPerRank` int32 region. | V2C/C2V loop protocol. | Before GMM2 group loops. | GMM2 reads at group start; CombineV2 reads at group start. | None separate from normal path cleanup. | GMM2-only computes or copies cumsum from external counts; official source path not fully reproduced. |
| `preSumBeforeRank` | Official cross-rank prefix path. | `BlockEpilogue2` final D placement. | Workspace binding in `initBuffer`. | `workspaceInfo.ptrSumBeforeRank`; after debug GMM regions. | `EP * expertPerRank` int32. | Shared routing state. | Before final epilogue writeback. | `BlockEpilogue2` reads per destination EP. | Normal path cleanup follows combine. | GMM2-only zeros it; acceptable only for EP=1 debug, not a general official lifecycle proof. |
| GMM2 AIC input tile state | Official `GMM2`. | `BlockMmad`. | Per group in `GMM2`, after V2C wait. | `gmA2I4_I8`, W2 `ptrB2`, scale `ptrScale2`, output `gmC2`. | `L1TileShape` scheduler; M doubled for INT4. | `SYNCFLAGV2C`. | AIV sets after hidden/scale production. | AIC waits before group tiles. | `BlockMmad::Finalize`. | Uses official `GMM2`, but driven by manually seeded state. |
| GMM2 accumulator / D2 region | `BlockMmad` and Fixpipe. | `BlockEpilogue2`. | Inside `GMM2` tile loop. | `gmC2` / debug `ptrCGMM2`; D2 high/low layout from `layoutD2`. | High/low halves separated by `n2` in `BlockEpilogue2`. | C2V. | AIC finalizes per sync group. | CombineV2 waits C2V before epilogue. | CombineV2 drains remaining flags. | Diagnostic taps read int32 accumulator and actual D2, but lifecycle source remains debug-only. |
| C2V handoff state | `BlockMmad::Finalize(..., SYNCFLAGC2V)`. | `CombineV2`. | GMM2 group completion. | Cross-core flag state. | Per sync group. | `SYNCFLAGC2V`. | After AIC GMM2 group tiles finish. | `CombineV2` waits before `BlockEpilogue2`. | CombineV2 drains all groups. | AIV side uses synthetic ready helper in GMM2-only path; official source must be preserved in next fix. |
| `BlockEpilogue2` input state | Official D2 high/low, `gmPerTokenScale2`, MAux2, token state. | `BlockEpilogue2`. | Constructed before combine. | `gmC2`, `gmCGMM2`, scale GM, MAux2, shmem `offsetD`. | 32-row AIV split; high row offset and low row offset separated by `n2`. | UB events plus C2V waits. | After C2V wait. | Event waits inside epilogue. | `BlockEpilogue2::Finalize`. | Formula reconstruction from actual D2 is exact, but official lifecycle equivalence is not proven. |
| FP32 post-dequant debug tap | `BlockEpilogue2` W4A8_DEBUG path. | Host probe readback. | During normal `BlockEpilogue2`. | `ptrDebugGMM2` override or workspace `ptrCGMM2`. | `maxOutputSize * n2` float. | `EVENT_ID7` inside debug copy. | After high*16+low, aux, hidden scale. | Host sync after op completion. | Epilogue finalize. | Latest tap is finite/nonzero but fails strict reference tolerance; not accepted as Stage 2.2 pass. |

Confirmed deviation before any next behavioral patch:

- The current isolated GMM2-only path reuses the official `GMM2`, `CombineV2`, `BlockEpilogue2`, packed W2 access,
  MMAD, Fixpipe, and D2/AIV formula, but it does not yet prove the complete official producer/consumer lifecycle.
  Specifically, token state and packed hidden/scale state are manually seeded, and AIV readiness is synthetically
  signaled instead of coming only from the successful official `DispatchAndCombine` producer loop.

Next allowed correction strategy:

- Start from the successful official `DispatchAndCombine` W4A8 path and preserve official AIC/AIV roles,
  token/expert state, V2C/C2V protocol, GMM2 loop state, packed W2 access, D2 source region, `BlockEpilogue2`,
  `CombineV2`, and final drain.
- Modify only the hidden input boundary by overlaying the already validated Stage 2.1 packed hidden INT4 tensor and
  per-token hidden scale at the same official point that `DispatchAndCombine` currently overlays external GMM2
  hidden/scale.
- Before changing behavior, complete Gate A row-identity evidence: routed token set, routed-row order, active expert
  set, `expert_token_nums`, prefix sums, expert-local row starts/offsets, active-row count, and padded-row
  interpretation.

## Stage 2.2 Direct Accumulator-to-D2 Diagnostic - 2026-06-27T01:59Z

Historical section. It superseded the `2026-06-27T01:52Z` actual-D2 reconstruction
handoff by adding and validating a direct comparator from the official int32 accumulator readback to the actual
raw high-D2 and low-D2 readbacks. No kernel behavior, lifecycle flag, V2C/C2V protocol, workspace state, tolerance,
production host tiling, public grouped-matmul path, packed-weight format, or scale formula was changed.

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A and int32 accumulator Gate B pass for top-1 expert 0; Gate C remains finite/nonzero but fails strict max-abs tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate C strict numerical match and unresolved Fixpipe/D2 value contract. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No host tiling enablement. |

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`

Exact behavior implemented:

- Added `gmm2.actual_d2_from_accumulator_diagnostics` to the normal Stage 2.2 Gate C summary.
- The diagnostic compares actual raw high-D2 and low-D2 readbacks against variants built directly from the official
  `gmm2_accumulator_int32` readback and packed scale words.
- It also records low32 effective scale-ratio statistics for accumulator entries above absolute thresholds `1`,
  `16`, and `128`.
- The diagnostic is comparator-only. It does not change `passed`, strict tolerances, the official kernel path, or
  production SVDQ fail-closed behavior.

Official source locations tied to this diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp:454`:
  debug copy of the int32 accumulator before Fixpipe.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp:461`:
  official Fixpipe writes the FP16 D2 region.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:200`:
  raw debug mode 2 reads the high FP16 D2 half.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:212`:
  raw debug mode 3 reads the low FP16 D2 half.

Validation commands:

- Syntax and whitespace checks:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  and `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- NPU/process preflight logs captured before validation.
- Real-device normal Gate C probe:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_d2_from_accumulator_true_top1_expert0.json`

Evidence paths:

- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_d2_from_accumulator_preflight_npus.log`
- Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_d2_from_accumulator_preflight_python.log`
- Active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_d2_from_accumulator_preflight_processes.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_d2_from_accumulator_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_d2_from_accumulator_true_top1_expert0.exitcode`
  contains `1`, expected because Stage 2.2 still fails strict Gate C.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_from_accumulator_true_top1_expert0.json`

Validated status fields from the new summary:

- `gate_a_input_boundary_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_post_dequant_reference_passed: false`
- `official_gmm2_numerical_gate_passed: false`

Numerical result:

- Strict current Gate C reference remains unchanged and fails max tolerance:
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.
- Actual-D2 reconstruction of the normal post-dequant tap remains exact:
  max abs `0.0`, mean abs `0.0`.
- Direct accumulator-to-D2 best variants:
  - high half: `scale_low32_fp16_toward_zero`, best key
    `[0.00048828125, 2.2813444957137108e-05, 1136]`.
  - low half: `scale_low32_fp16_toward_zero`, best key
    `[0.001953125, 7.748615462332964e-05, 5221]`.
- Effective low32 scale-ratio stats for `abs(accumulator) >= 128` are close to the stored low32 scale but not exact:
  - high half ratio-minus-scale abs mean `1.1146955785079626e-06`, abs max `4.906207323074341e-06`.
  - low half ratio-minus-scale abs mean `1.1521455007823533e-06`, abs max `5.553476512432098e-06`.

Current interpretation:

1. Low32 scale with toward-zero-like FP16 behavior is still the closest host approximation for both actual D2 halves.
2. The raw high/low D2 values cannot be reproduced within the strict D2 tolerance from a simple
   `fp16(accumulator * low32_scale)` model, even though the effective scale ratios are close.
3. The AIV formula from actual D2 to post-dequant is exact, so the remaining mismatch is specifically in the
   official Fixpipe `VDEQF16`/D2 value contract from int32 accumulator plus scale buffer to FP16 D2.
4. Stage 2.2 remains `FAIL / IN PROGRESS`. Do not relax the gate, modify lifecycle state, or proceed to Stage 2.3.

## Stage 2.2 Actual D2 to AIV Post-Dequant Reconstruction - 2026-06-27T01:52Z

Historical section. It superseded the `2026-06-27T01:46Z` mixed-rounding diagnostic
by adding and validating an actual-D2 readback reconstruction of the normal W4A8_DEBUG post-dequant tap. No kernel
behavior, lifecycle flag, V2C/C2V protocol, workspace state, tolerance, production host tiling, public
grouped-matmul path, packed-weight format, or scale formula was changed.

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A and int32 accumulator Gate B pass for top-1 expert 0; Gate C remains finite/nonzero but fails strict max-abs tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate C strict numerical match and unresolved Fixpipe/D2 value contract. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No host tiling enablement. |

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`

Exact behavior implemented:

- Added `gmm2.actual_d2_post_dequant_reconstruction` to the normal Stage 2.2 Gate C summary.
- The normal probe now relaunches the same official debug op with `swigluLimit=451000.0` and `453000.0` to capture
  official `BlockEpilogue2` raw high-D2 and low-D2 readbacks for the same validated hidden boundary.
- The diagnostic reconstructs the normal post-dequant tap from those actual D2 readbacks using the official AIV
  formula: `high * 16 + low + aux`, then hidden-scale multiplication.
- The diagnostic is comparator-only. It does not change `passed`, strict tolerances, the official kernel path, or
  production SVDQ fail-closed behavior.

Official source locations tied to this diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:200`:
  raw debug mode 2 writes the high FP16 D2 half after FP32 cast.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:212`:
  raw debug mode 3 writes the low FP16 D2 half after FP32 cast.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:224-262`:
  normal `BlockEpilogue2` applies high*16+low, aux, hidden scale, and writes the W4A8_DEBUG FP32 tap.

Validation commands:

- Syntax check:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- Diff whitespace check:
  `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- NPU/process preflight logs captured before validation.
- Real-device normal Gate C probe:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_actual_d2_reconstruct_true_top1_expert0.json`

Evidence paths:

- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_actual_d2_reconstruct_preflight_npus.log`
- Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_actual_d2_reconstruct_preflight_python.log`
- Active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_actual_d2_reconstruct_preflight_processes.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_actual_d2_reconstruct_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_actual_d2_reconstruct_true_top1_expert0.exitcode`
  contains `1`, expected because Stage 2.2 still fails strict Gate C.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_actual_d2_reconstruct_true_top1_expert0.json`

Validated status fields from the new summary:

- `gate_a_input_boundary_passed: true`
- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_c2v_handoff_verified: true`
- `official_gmm2_post_dequant_finite: true`
- `official_gmm2_post_dequant_nonzero: true`
- `official_gmm2_post_dequant_reference_passed: false`
- `official_gmm2_numerical_gate_passed: false`

Numerical result:

- Strict current Gate C reference remains unchanged and fails max tolerance:
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.
- `actual_d2_post_dequant_reconstruction.enabled: true`.
- Actual-D2 reconstruction of the normal post-dequant tap passes exactly:
  max abs `0.0`, mean abs `0.0`, failed elements `0`.
- Prior accumulator-derived best variants remain unchanged:
  `scale_low32_fp16_toward_zero` and `high_toward_zero__low_toward_zero` both have best key
  `[0.00023069977760314941, 1.006269667414017e-05, 20]`.

Current interpretation:

1. The official AIV `BlockEpilogue2` high*16+low, aux addition, hidden-scale multiplication, and debug copy are
   correctly modeled when starting from actual D2 high/low readbacks.
2. The remaining mismatch is not in the AIV post-dequant formula or the high/low row mapping.
3. The unresolved boundary is now the exact official Fixpipe `VDEQF16`/D2 value contract between the exact int32
   accumulator readback and the actual high/low D2 FP16 values.
4. Stage 2.2 remains `FAIL / IN PROGRESS`. Do not relax the gate, modify lifecycle state, or proceed to Stage 2.3.

## Stage 2.2 Mixed High/Low D2 Rounding Diagnostic - 2026-06-27T01:46Z

Historical section. It superseded the `2026-06-27` official-path constraint
handoff by adding and validating a diagnostic-only comparator for independent high-half and low-half FP16 D2
rounding candidates. No kernel behavior, lifecycle flag, V2C/C2V protocol, workspace state, tolerance, production
host tiling, public grouped-matmul path, packed-weight format, or scale formula was changed.

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A and int32 accumulator Gate B pass for top-1 expert 0; Gate C remains finite/nonzero but fails strict max-abs tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate C strict numerical match and unresolved Fixpipe/D2/AIV contract detail. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No host tiling enablement. |

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`

Exact behavior implemented:

- Added a compact host-side `low32_mixed_high_low_rounding` diagnostic under
  `gmm2.post_dequant_variant_diagnostics`.
- The diagnostic starts from the official `gmm2_accumulator_int32` readback, uses the already identified plausible
  low32 packed scale word, and applies independent FP16 D2 storage rounding candidates to the high and low
  accumulator halves before the official `BlockEpilogue2` formula:
  `high * 16 + low + aux`, then hidden-scale multiplication.
- The diagnostic is comparator-only. It does not change `passed`, strict tolerances, the official kernel path, or
  production SVDQ fail-closed behavior.

Official source locations tied to this diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp:446-461`:
  official GMM2 copies scale into the Fixpipe buffer and writes FP16 D2 through `copyL0CToGm`.
- `csrc/mc2/dispatch_ffn_combine/op_kernel/utils/copy_l0c_to_gm_custom.hpp:10-43`:
  per-channel row-major Fixpipe call used by the W4A8 block path.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:196-262`:
  `BlockEpilogue2` casts high/low D2 halves to FP32, applies `high * 16 + low`, aux, hidden scale, and the
  W4A8_DEBUG FP32 tap.

Validation commands:

- Syntax check:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- NPU/process preflight logs captured before validation.
- Real-device normal Gate C probe:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_mixed_rounding_true_top1_expert0.json`

Evidence paths:

- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_mixed_rounding_preflight_npus.log`
- Python logical-device preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_mixed_rounding_preflight_python.log`
- Active process preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_mixed_rounding_preflight_processes.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_mixed_rounding_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_mixed_rounding_true_top1_expert0.exitcode`
  contains `1`, expected because Stage 2.2 still fails strict Gate C.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_mixed_rounding_true_top1_expert0.json`

Validated status fields from the new summary:

- `official_gmm2_entry_reached: true`
- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_c2v_handoff_verified: true`
- `official_gmm2_post_dequant_finite: true`
- `official_gmm2_post_dequant_nonzero: true`
- `official_gmm2_post_dequant_reference_passed: false`
- `official_gmm2_numerical_gate_passed: false`
- `gate_a_input_boundary_passed: true`

Numerical result:

- Strict current Gate C reference remains unchanged and fails max tolerance:
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.
- Prior best single rounding variant remains:
  `scale_low32_fp16_toward_zero`, best key `[0.00023069977760314941, 1.006269667414017e-05, 20]`.
- New mixed high/low diagnostic best variant:
  `high_toward_zero__low_toward_zero`, best key
  `[0.00023069977760314941, 1.006269667414017e-05, 20]`.
- The mixed diagnostic therefore rejects the hypothesis that different high-half and low-half FP16 rounding modes
  explain the remaining mismatch.

Current interpretation:

1. Low32 scale remains the only plausible packed scale word for this path.
2. Independent high/low D2 rounding candidates do not improve beyond applying toward-zero to both halves.
3. Stage 2.2 still fails. The unresolved boundary remains the precise official Fixpipe `VDEQF16`/D2 behavior or a
   still-missing AIV comparator detail downstream of the exact int32 accumulator.
4. Do not relax the gate, modify lifecycle state, or proceed to Stage 2.3.

## Stage 2.2 Official GMM2 Path Constraint Reaffirmed - 2026-06-27

Historical section. It applied
`/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` as a binding
constraint for the next work. The earlier sections below are historical evidence. The appendix's all-zero
post-dequant symptom is superseded by the latest local probe evidence: Gate B int32 accumulator and Gate C
post-dequant readbacks are finite/nonzero, but Stage 2.2 still fails strict numerical comparison.

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A and int32 accumulator Gate B pass for top-1 expert 0; Gate C remains finite/nonzero but fails strict max-abs tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate C strict numerical match and unresolved Fixpipe/D2/AIV contract detail. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No host tiling enablement. |

Binding requirements for the next patch:

- Do not use, modify, reinterpret, or debug public `torch_npu.npu_grouped_matmul`.
- Do not patch V2C/C2V flags, token state, prefix sums, workspace offsets, producer/consumer roles, D2 source
  addresses, or synchronization unless the deviation is first recorded against the successful official path.
- Preserve the successful official W4A8 lifecycle and substitute only the already validated Stage 2.1 packed hidden
  tensor and hidden scale at the official GMM2 input boundary.
- Keep `DispatchFFNCombineW4A8SVDQ` production tiling fail-closed until real-device Stage 2.2 and later numerical
  gates pass.
- Treat compilation, registration, source inspection, loop entry, and status manifests as provenance only, not gate
  progress.

Required pre-patch state table:

- The source-backed official-vs-debug lifecycle table required by the appendix is recorded in the historical section
  `Stage 2.2 Official-Path Appendix Applied - 2026-06-27T02:05Z`.
- That table cites the official flow through
  `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`,
  `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h`,
  `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp`, and
  `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp`.
- The next behavioral patch must update that table first with the exact deviation being corrected.

Latest numerical evidence remains:

- Gate A input boundary passed.
- Gate B int32 accumulator passed exactly with mismatch count `0` and nonzero output.
- C2V handoff is verified by finite/nonzero AIV consumption.
- Gate C post-dequant output is finite/nonzero but fails strict max tolerance:
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.
- The diagnostic-only post-dequant variant comparator identifies low32 scale bits as plausible and high32 scale bits
  as wrong; `scale_low32_fp16_toward_zero` is closer but still fails strict max tolerance.

Next unresolved boundary:

- Continue only at the official Fixpipe `VDEQF16`/D2 storage and `BlockEpilogue2`/AIV comparison contract.
- Do not advance to SVDQ down composition, final combine, or production fused-op enablement while Stage 2.2 remains
  `FAIL / IN PROGRESS`.

## Stage 2.2 Post-Dequant Variant Diagnostic Added - 2026-06-27T02:35Z

Historical section. It superseded the `2026-06-27T02:20Z` status-field section by adding a diagnostic-only
post-dequant variant comparator built from the official GMM2 int32 accumulator readback. No kernel behavior,
lifecycle flag, V2C/C2V protocol, workspace state, tolerance, production host tiling, public grouped-matmul path,
scale formula, or packed-weight format was changed.

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A and int32 accumulator Gate B pass for top-1 expert 0; Gate C remains finite/nonzero but fails strict max-abs tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate C strict numerical match and unresolved D2/Fixpipe/AIV contract detail. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No host tiling enablement. |

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`

Exact behavior implemented:

- Added `gmm2.post_dequant_variant_diagnostics` to the normal Stage 2.2 Gate C summary.
- The diagnostic starts from the official `gmm2_accumulator_int32` readback, then applies packed scale-bit variants,
  FP16 D2 storage rounding candidates, W4A8 aux bias, and hidden per-token scale before comparing with the official
  W4A8_DEBUG post-dequant AIV tap.
- The diagnostic is explicitly marked as comparator-only. It does not change `passed`, strict tolerances, or the
  official kernel path.

Official source locations tied to this diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:218-231`:
  `QuantTileCopy` selects `CopyL0CToGm<..., PER_CHANNEL>`.
- `csrc/mc2/dispatch_ffn_combine/op_kernel/utils/copy_l0c_to_gm_custom.hpp:10-43`:
  the per-channel row-major specialization sets `FixpipeParamsV220`, including `quantPre`, then calls
  `AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:224-262`:
  `BlockEpilogue2` performs high*16+low, aux addition, hidden-scale multiplication, and W4A8_DEBUG FP32 tap.

Validation commands:

- Syntax check:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- Real-device normal Gate C probe:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_post_dequant_variants_true_top1_expert0.json`

Evidence paths:

- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_post_dequant_variants_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_post_dequant_variants_true_top1_expert0.exitcode`
  contains `1`, expected because Stage 2.2 still fails strict Gate C.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_post_dequant_variants_true_top1_expert0.json`

Validated status fields from the new summary:

- `gate_a_input_boundary_passed: true`
- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_c2v_handoff_verified: true`
- `official_gmm2_post_dequant_finite: true`
- `official_gmm2_post_dequant_nonzero: true`
- `official_gmm2_post_dequant_reference_passed: false`
- `official_gmm2_numerical_gate_passed: false`

Numerical result:

- Strict current Gate C reference remains unchanged and fails max tolerance:
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.
- `post_dequant_variant_diagnostics.enabled: true`.
- Best diagnostic variant:
  `scale_low32_fp16_toward_zero`, best key `[0.00023069977760314941, 1.006269667414017e-05, 20]`.
- The current nearest-even reference matches the old strict error:
  `scale_low32_fp16_nearest_even` max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.
- High32 scale variants are rejected by magnitude:
  max abs `0.3963119685649872`, mean abs `0.05269600450992584`.

Current interpretation:

1. The post-dequant comparator now proves the official packed scale word is the low 32 bits; high 32 bits are not
   plausible for this path.
2. A simple FP16 `toward_zero` candidate is closer than nearest-even but still fails strict max tolerance
   (`0.00023069977760314941 > 0.0002`), so Stage 2.2 still fails.
3. The unresolved boundary is narrower: exact int32 accumulator plus low32 scale is correct; the remaining mismatch
   is in the precise Fixpipe VDEQF16/FP16 storage behavior or in a still-missing detail of the AIV post-dequant
   comparator. Do not relax the gate or move to Stage 2.3.

## Stage 2.2 Required Status Fields Added to Gate C Probe - 2026-06-27T02:20Z

Historical section. It superseded the `2026-06-27T02:05Z` report-only appendix application by adding and validating
the appendix-required status fields in the normal Stage 2.2 post-dequant probe. No kernel behavior, tolerance,
scale formula, packing formula, public grouped-matmul path, lifecycle flag, or production SVDQ path was changed.

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A and int32 accumulator Gate B pass for top-1 expert 0; Gate C remains finite/nonzero but fails strict max-abs tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate C strict numerical match and unresolved D2/Fixpipe/AIV contract detail. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No host tiling enablement. |

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`

Exact behavior implemented:

- The normal post-dequant Stage 2.2 probe now reports the appendix-required fields:
  `official_gmm2_entry_reached`, `official_gmm2_loop_count`, `official_gmm2_active_tile_count`,
  `official_gmm2_aic_raw_output_finite`, `official_gmm2_aic_raw_output_nonzero`,
  `official_gmm2_aic_reference_passed`, `official_gmm2_c2v_handoff_verified`,
  `official_gmm2_post_dequant_finite`, `official_gmm2_post_dequant_nonzero`,
  `official_gmm2_post_dequant_reference_passed`, and `official_gmm2_numerical_gate_passed`.
- The normal post-dequant summary now also records `gate_a_input_boundary_passed` and an
  `int32_accumulator_readback_reference` block, so the normal Gate C run carries the same Gate B accumulator proof
  that the raw-D2 diagnostic already exposed.
- The top-level `passed` decision now depends on Gate A, the int32 accumulator Gate B proof, finite/nonzero
  post-dequant output, and the strict Gate C reference comparison. Because Gate C still fails strict max-abs
  tolerance, the probe correctly remains failed.

Validation commands:

- Syntax check:
  `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- Four-visible-NPU preflight with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` showed four logical `Ascend910B4` devices.
- Real-device normal Gate C probe:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --summary-name phase_stage2_gmm2_status_fields_true_top1_expert0.json`

Evidence paths:

- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_status_fields_true_top1_expert0.log`
- Probe exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_status_fields_true_top1_expert0.exitcode`
  contains `1`, expected because Stage 2.2 still fails strict Gate C.
- Probe summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_status_fields_true_top1_expert0.json`

Validated status fields from the new summary:

- `gate_a_input_boundary_passed: true`
- `official_gmm2_entry_reached: true`
- `official_gmm2_loop_count: null`
- `official_gmm2_active_tile_count: null`
- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: true`
- `official_gmm2_accumulator_int32_reference_passed: true`
- `official_gmm2_c2v_handoff_verified: true`
- `official_gmm2_post_dequant_finite: true`
- `official_gmm2_post_dequant_nonzero: true`
- `official_gmm2_post_dequant_reference_passed: false`
- `official_gmm2_numerical_gate_passed: false`

Numerical result:

- Gate B int32 accumulator remains exact and nonzero:
  `int32_accumulator_readback_reference.passed: true`, `actual_nonzero: true`, mismatch count `0`.
- Gate C remains finite/nonzero but fails the strict max-abs tolerance:
  max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`,
  tolerance max abs `0.0002`, mean abs `2e-05`.

Current interpretation:

1. The probe evidence schema now satisfies the appendix status-field requirement for the normal Gate C run.
2. Stage 2.2 still fails. The unresolved boundary is still downstream of the exact int32 accumulator, in the
   official D2/Fixpipe/AIV post-dequant contract or the host comparator for that contract.
3. The next behavioral patch must continue from the official Fixpipe/D2/AIV contract. Do not alter lifecycle flags,
   token state, workspace state, or production `DispatchFFNCombineW4A8SVDQ`.

## Stage 2.2 Official-Path Appendix Applied - 2026-06-27T02:05Z

Historical section. It applied
`/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` as a binding
constraint update. The appendix's all-zero post-dequant description is historical for the earlier failing state:
the current local evidence after the host ABI repair is finite/nonzero Gate B accumulator and finite/nonzero Gate C
post-dequant readback, but Stage 2.2 still fails strict numerical comparison. Stage 2.2 therefore remains
`FAIL / IN PROGRESS`; no later stage or production path is unblocked.

| Stage | Status | Current gate |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden and hidden scale read back exactly in Stage 2.2 diagnostics. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate B int32 accumulator passes for top-1 expert 0; raw D2 and Gate C still fail strict numerical gates. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B raw-D2/Fixpipe and Gate C reference match. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No host tiling enablement. |

New stop condition before the next behavioral patch:

- Do not patch V2C/C2V flags, token state, prefix sums, workspace offsets, producer/consumer roles, D2 source
  addresses, or synchronization unless the deviation is identified in the table below and tied to the official
  successful path.
- Do not use public `torch_npu.npu_grouped_matmul`.
- Do not relax tolerances, repack weights, guess scale formulas, or proceed to SVDQ down/final combine.
- Treat compilation, registration, installation, source inspection, and loop entry as provenance only, not numerical
  gate progress.

Official source locations inspected for this update:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:220`: debug wrapper
  `InitGMM2OnlyFromPacked` delegates through the normal `Init` path and currently sets `gmm2OnlyFromPacked_ = false`
  at line `233`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:250`: AIC would enter
  `GMM2OnlyFromPacked` only when `params.gmm2OnlyFromPacked` is true; current debug path does not take that branch.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:271`: AIV would enter
  `GMM2OnlyDequantReadback` only when `params.gmm2OnlyFromPacked` is true; current debug path instead uses
  `DispatchAndCombine`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1199`: official
  `DispatchAndCombine` constructs routing, token/expert state, cumsum, hidden packing, V2C signaling,
  `BlockEpilogue2`, `CombineV2`, final drains, and unpermute.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:687`: official GMM2 AIC
  consumes `gmA2I4`, W2, scale2, writes `gmC2`, and optionally mirrors int32 accumulators to
  `ptrDebugGMM2Accumulator`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp:438`: official Fixpipe copies scale to
  Fixpipe buffer, optionally emits int32 accumulator debug at line `457`, and writes scaled FP16 D2 via
  `copyL0CToGm` at line `461`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1448`: `CombineV2` waits on
  GMM2 C2V flags and calls `BlockEpilogue2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:147`:
  `BlockEpilogue2` reads high/low FP16 D2 halves, combines them, adds W4A8 aux, multiplies hidden per-token scale,
  writes the FP32 W4A8_DEBUG tap, casts to output dtype, and writes peer memory.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/svdqw4_a8_gmm2_debug_readback.cpp:16`: debug kernel entry calls
  the wrapper with external packed hidden, external hidden scale, debug post-dequant, hidden readbacks, and int32
  accumulator readback.
- `csrc/mc2/svdq_w4a8_gmm2_debug_readback/svdq_w4a8_gmm2_debug_readback_torch_adpt.h:119`: torch adapter allocates
  the debug outputs and returns `(gmm2_post_dequant, hidden_x_readback, hidden_scale_readback,
  gmm2_accumulator_int32)`.

Official-vs-debug lifecycle table:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | Full path: `BlockEpilogue1` writes packed hidden at `dispatch_ffn_combine_w4_a8_kernel.hpp:1358`. Debug override: external packed hidden is copied over the same region at `1364-1376`. | GMM2 AIC consumes `gmA2I4[...]` at `dispatch_ffn_combine_w4_a8_kernel.hpp:773`. Debug readback copies from `gmA2I4_I8` at `1391`. | `initBuffer` binds `gmA2I4`/`gmA2I4_I8` to `workspaceInfo.ptrA2Int4` at `dispatch_ffn_combine_w4_a8_kernel.hpp:293-300`. | `WorkspaceInfo` sets `ptrA2Int4 = params.ptrWorkspace + workspaceOffset` and reserves `maxOutputSize * k2` bytes at `1597-1602`. Full-path override uses `gmOffsetD = params.layoutD1.GetOffset(offsetC)` at `1356`. | Logical `layoutD1{maxOutputSize, k2}` from `dispatch_ffn_combine_w4_a8.h:319`; GMM2 uses `layoutA2.GetTileLayout` at `dispatch_ffn_combine_w4_a8_kernel.hpp:735` with `currentM * 2`. | V2C `SYNCFLAGV2C`. | Official full path sets V2C after hidden packing/override at `dispatch_ffn_combine_w4_a8_kernel.hpp:1380-1382`. | GMM2 waits at group start or sync boundary at `dispatch_ffn_combine_w4_a8_kernel.hpp:745-746`. | Hidden debug readback is copied after `BlockEpilogue1.Finalize()` at `1384-1401`. | Current debug path preserves full lifecycle and overrides only the official packed-hidden boundary. It does not enter the manual `GMM2OnlyFromPacked` branch because `gmm2OnlyFromPacked_` is false. Gate A packed readback is exact. |
| hidden scale `gmPerTokenScale2` | Full path: `BlockEpilogue1` writes per-token scale through `gmPerTokenScale2[rowStartThisCore]` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1359`. Debug override copies external scale at `1372-1376`. | `BlockEpilogue2` consumes `gmPerTokenScale` per row at `block_epilogue_w4a8post_pertoken_v2.hpp:248-254`. Debug readback copies at `dispatch_ffn_combine_w4_a8_kernel.hpp:1394-1399`. | `initBuffer` binds `gmPerTokenScale2` to `workspaceInfo.ptrPerTokenScale2` at `dispatch_ffn_combine_w4_a8_kernel.hpp:307-310`. | `WorkspaceInfo` sets `ptrPerTokenScale2` after `ptrPerTokenScale` and reserves `maxOutputSize * sizeof(float)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1567-1572`. | One FP32 scale per packed hidden row; `BlockEpilogue2` uses `gmScaleOffset = (preSrcExpertSum + blockCoord.m()) / 2` at `block_epilogue_w4a8post_pertoken_v2.hpp:248`. | V2C `SYNCFLAGV2C`. | Same as packed hidden: signal after full-path hidden production/override at `1380-1382`. | GMM2 waits at `745-746`; `CombineV2` waits on C2V before dequant at `1500-1502`. | Debug readback copies full `maxOutputSize` scale buffer at `1394-1399`. | Current debug path preserves official scale consumer and overrides only the producer result. Gate A scale readback is exact. |
| `tokenPerExpert` | Official routing writes local token counts via `moe_init_routing_quant_v2` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1211-1216`, then all-gather/prefix code updates peer token state at `1048-1126`. | `GetCumsumForMMAIV` consumes it at `1237-1239`; `BlockEpilogue2` consumes rank/expert slices at `block_epilogue_w4a8post_pertoken_v2.hpp:277-279`. | `initBuffer` binds `tokenPerExpert` to peermem `offsetPeerTokenPerExpert` and builds `tokenPerExpertLayout` at `dispatch_ffn_combine_w4_a8_kernel.hpp:312-316`. | Peermem base is `shmem() + peermemInfo.offsetPeerTokenPerExpert`; `PeermemInfo` sets the offset at `dispatch_ffn_combine_w4_a8_kernel.hpp:1656-1659`. | `Layout3D(AlignUp(EP * expertPerRank + 1, 128), expertPerRank)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:315-316`. | Internal peer-memory sync and cache-line wait in `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2`. | Full path initializes routing before any GMM work at `1208-1239`. | Prefix/all-gather waits for peer token rows with `gm_signal_wait_until_ne` at `1090-1094`. | Reset after combine at `1412-1415`. | Current debug path passes `externalExpertTokenNums`, but the active path does not seed token state from it; `SeedGMM2OnlyTokenState` at `944-960` is inactive because `gmm2OnlyFromPacked_` is false. Routing identity still must be reported row-by-row per the appendix. |
| `cumsumMM` | Official producer is `GetCumsumForMMAIV(tokenPerExpert, cumsumMM, ...)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1237-1239`. | GMM1/GMM2/CombineV2 read per-expert active rows at `602`, `714`, and `1467`. | `initBuffer` binds `cumsumMM` to `workspaceInfo.ptrcumsumMM` at `dispatch_ffn_combine_w4_a8_kernel.hpp:287`. | `WorkspaceInfo` places `ptrcumsumMM` after expanded row indices at `dispatch_ffn_combine_w4_a8_kernel.hpp:1559-1564`. | `EP * expertPerRank` int32 prefix matrix; GMM2 uses final EP row `(EP - 1) * expertPerRank + groupIdx`. | Routing/state sync before GMM. | Producer runs before GMM flags are released at `1237-1257`. | GMM2 reads after V2C waits at `745-746`. | Expert token output copies final cumsum row at `1251-1253`. | Current debug path uses official full-path cumsum, not the inactive manual GMM2-only seed. Gate A has enough counts for the top-1 expert 0 diagnostic, but the appendix requires fuller routed-row/padded-row evidence. |
| `preSumBeforeRank` | Official producer is `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2`, especially `DataCopyPad(preSumBeforeRank...)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1118-1122`. | `BlockEpilogue2` uses `preSumBeforeRank(dstEpIdx * expertPerRank + groupIdx)` to calculate peer output offsets at `block_epilogue_w4a8post_pertoken_v2.hpp:277-307`. | `initBuffer` binds `preSumBeforeRank` to `workspaceInfo.ptrSumBeforeRank` at `dispatch_ffn_combine_w4_a8_kernel.hpp:317`. | `WorkspaceInfo` places `ptrSumBeforeRank` after the debug GMM buffers at `dispatch_ffn_combine_w4_a8_kernel.hpp:1628-1630`. | `EP * expertPerRank` int32 values. | Peer token sync. | Built before GMM/Combine at `1235-1247`. | Consumed inside `BlockEpilogue2` after `CombineV2` waits on C2V at `1500-1502`. | No separate final drain beyond `BlockEpilogue2.Finalize()` and token reset. | Current debug path uses official producer. No patch is allowed to zero or seed this field unless a source-backed deviation is identified. |
| GMM2 AIC input tile state | Official GMM2 constructs `inGroupProblemShape{currentM * 2, n2, k2}`, layouts, W2/scale2 pointers, and scheduler state at `dispatch_ffn_combine_w4_a8_kernel.hpp:687-740`. | `blockMmad` consumes `gmA2I4`, W2, and scale2 at `dispatch_ffn_combine_w4_a8_kernel.hpp:773-776`. | `GMM2(params)` initializes scheduler/core-loop state locally at `687-705`; W2/scale2 global buffers are selected per group at `725-729`. | `gmOffsetA`, `gmOffsetB`, `gmOffsetC`, and `gmOffsetS` are calculated at `dispatch_ffn_combine_w4_a8_kernel.hpp:761-764`. | CATLASS block scheduler over `L1TileShape`; `startCoreIdx` rotates at `788`. | V2C before GMM2 tile group; C2V after Fixpipe. | GMM2 waits on V2C at `745-746` before the group tile loop. | `CombineV2` waits on C2V at `1500-1502`. | `blockMmad.SynchronizeBlock()` and `Finalize(params.expertPerRank - 1, 0)` at `790-793`. | Current Gate B int32 accumulator is exact/nonzero for top-1 expert 0, proving this tile producer and packed input/weight access for that diagnostic. |
| GMM2 accumulator / D2 region | Official producer is `BlockMmad` L0C accumulation plus Fixpipe at `block_mmad_w4a4.hpp:438-461`; debug int32 accumulator tap is at `454-458`. | `BlockEpilogue2` reads high and low FP16 D2 halves from `gmC2` at `block_epilogue_w4a8post_pertoken_v2.hpp:157-184`. | `WorkspaceInfo` binds `gmC2` at `dispatch_ffn_combine_w4_a8_kernel.hpp:305` and allocates `maxOutputSize * n2 * sizeof(ElementC) * 2` at `1583-1587`. | `GMM2` writes `gmC2[gmGroupOffsetC + gmOffsetC]` at `dispatch_ffn_combine_w4_a8_kernel.hpp:773-775`; `BlockEpilogue2` reads `gmCOffsetH` and `gmCOffsetL = + n2` at `block_epilogue_w4a8post_pertoken_v2.hpp:157-160`. | D2 stores doubled rows/high-low halves; `BlockEpilogue2` uses `layoutGM2{actualM / 2, N, n2 * 2}` and UB `n0` at `177-180`. | C2V via `BlockMmad::Finalize`. | Fixpipe runs after K-loop last at `block_mmad_w4a4.hpp:438-461`; finalize emits the group flag at `485`. | `CombineV2` waits C2V at `dispatch_ffn_combine_w4_a8_kernel.hpp:1499-1502`. | `BlockEpilogue2.Finalize()` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1525`. | Current int32 accumulator exact mismatch count is `0`. Raw high/low FP16 D2 readbacks are finite/nonzero but miss strict reference by up to 2 FP16 ULP, so the unresolved boundary is Fixpipe/D2 rounding or the raw-D2 comparator, not AIC int32 accumulation. |
| C2V handoff state | Official producer is `blockMmad.Finalize(...)` after Fixpipe at `block_mmad_w4a4.hpp:484-485`; GMM2 also finalizes at `dispatch_ffn_combine_w4_a8_kernel.hpp:790-793`. | `CombineV2` waits flags before each group/tile at `dispatch_ffn_combine_w4_a8_kernel.hpp:1499-1502`. | `BlockEpilogue2.InitFlag()` initializes local epilogue flags at `dispatch_ffn_combine_w4_a8_kernel.hpp:1450` and `block_epilogue_w4a8post_pertoken_v2.hpp:121-130`. | Cross-core flag IDs are `SYNCFLAGC2V = 9` and `SYNCFLAGV2C = 10` at `dispatch_ffn_combine_w4_a8_kernel.hpp:45-46`; `BlockMmad` receives `syncLoopIdx` at `773-776`. | One C2V wait group per expert sync task. | C2V `SYNCFLAGC2V`. | Produced after Fixpipe completes for the group/tile. | Waited immediately before `BlockEpilogue2` work in `CombineV2`. | `BlockEpilogue2.Finalize()` drains UB/MTE flags at `block_epilogue_w4a8post_pertoken_v2.hpp:132-141`. | Current debug path should preserve official C2V timing because `gmm2OnlyFromPacked_` is false. Do not add or move flags without a source-backed deviation. |
| `BlockEpilogue2` input state | Official input is `gmCGMM2`, `gmC2`, `gmPerTokenScale2`, `ptrMAux2`, `realTileCoord`, `realTileShape`, `groupIdx`, `preSrcExpertSum`, and `preSumBeforeRank` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1511-1512`. | `BlockEpilogue2` consumes high/low D2, aux, hidden scale, and peer offset state at `block_epilogue_w4a8post_pertoken_v2.hpp:147-313`. | Full path constructs `BlockEpilogue2::Params` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1325-1339`. | Debug FP32 tap points `gmCGMM2` to `params.ptrDebugGMM2` when provided at `dispatch_ffn_combine_w4_a8_kernel.hpp:1620-1627`. | Per tile: 32-row split into high/low packed halves, `n0` UB stride, `n2` GM stride. | C2V wait plus local UB/MTE event IDs. | Runs only after `CombineV2` C2V waits. | Uses local event waits in `block_epilogue_w4a8post_pertoken_v2.hpp:182-193`. | `Finalize()` waits UB and debug MTE3/V events at `132-141`. | Current normal post-dequant tap is finite/nonzero but max abs exceeds strict tolerance. Raw debug modes selected by `swigluLimit` (`450000-460000`) return D2-derived FP32 before full dequant for diagnostics only. |
| FP32 post-dequant debug tap | Official W4A8_DEBUG producer in `BlockEpilogue2` combines high/low halves, adds aux, multiplies hidden scale, then writes `gmGMM2` at `block_epilogue_w4a8post_pertoken_v2.hpp:224-262`. | Host probe consumes returned `gmm2_post_dequant`; production peer output is written after BF16/FP16 cast at `266-307`. | `ptrCGMM2` is set to `params.ptrDebugGMM2` when provided at `dispatch_ffn_combine_w4_a8_kernel.hpp:1620-1627`. | Debug output uses `gmCOffset = (preSrcExpertSum * n2 + blockCoord.m() * n2) / 2 + blockCoord.n()` at `block_epilogue_w4a8post_pertoken_v2.hpp:170-172`. | Output logical shape is `[maxOutputSize, hidden_size]` FP32 in the torch adapter at `svdq_w4a8_gmm2_debug_readback_torch_adpt.h:121`. | Debug copy uses EVENT_ID7 MTE3/V serialization. | Debug tap occurs before final `Cast<ElementD, float, CAST_RINT>` at `block_epilogue_w4a8post_pertoken_v2.hpp:258-266`. | Host observes it only after op completion. | `BlockEpilogue2.Finalize()` drains EVENT_ID7 at `139-141`. | Current Gate C is finite/nonzero with max abs `0.00037679076194763184` and mean abs `1.82786079676589e-05`; strict max tolerance still fails. |

Deviation summary before the next patch:

1. The active debug op is not taking the standalone `GMM2OnlyFromPacked` state machine despite the wrapper name;
   `gmm2OnlyFromPacked_ = false` means the current diagnostic uses the official full lifecycle and only overrides
   the hidden packed-input/scale boundary after `BlockEpilogue1`.
2. Gate A hidden and scale readbacks are exact, but the report still needs fuller routed-row identity and
   padded-row evidence for all required fields from the appendix.
3. Gate B int32 accumulator is exact/nonzero; raw D2 high/low FP16 values are finite/nonzero but fail strict
   reference by <=2 FP16 ULP.
4. Gate C FP32 post-dequant is finite/nonzero but fails strict max-abs tolerance. The next numerical investigation
   must stay at the official Fixpipe/D2/AIV contract boundary and must not alter lifecycle state speculatively.

## Stage 2.2 GMM2 Host ABI Fixed, Accumulator Gate Proven - 2026-06-27T01:20Z

This section is the latest authoritative handoff. It supersedes the `2026-06-27T01:05Z` host-extension ABI
handoff. The stale host-extension ABI/cache blocker is fixed in the current local build/install, and the official
GMM2 int32 accumulator readback now proves the AIC producer is nonzero and exact. Stage 2.2 is still not passed:
raw D2/Fixpipe half readback and final post-dequant comparison still miss the strict reference gate.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Hidden packed INT4 and hidden scale read back exactly in the new Stage 2.2 probes. |
| Stage 2.2 Gate A input boundary | PASS for top-1 expert 0 diagnostic | Canonical hidden, packed hidden, hidden scale, routing identity, padded rows, W2 metadata, and override readbacks are finite/nonzero/exact where required. |
| Stage 2.2 Gate B int32 accumulator | PASS for top-1 expert 0 diagnostic | `gmm2AccumulatorInt32` is returned, nonzero, and exact against the official-contract int32 reference with mismatch count `0`. |
| Stage 2.2 raw D2/Fixpipe half readback | FAIL / IN PROGRESS | High and low FP16 D2 half readbacks are finite/nonzero but fail strict raw-C2 reference tolerances with <=2 FP16 ULP differences. |
| Stage 2.2 Gate C post-dequant | FAIL / IN PROGRESS | Official W4A8_DEBUG post-dequant tap is finite/nonzero but fails strict reference max-abs tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 raw D2/Gate C numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints reaffirmed:

- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding.
- No public `torch_npu.npu_grouped_matmul` path was used, modified, reinterpreted, or debugged.
- No speculative scale formula, packed-weight repack, V2C/C2V lifecycle change, or production SVDQ path change was
  made.
- The official `dispatch_ffn_combine_w4_a8` implementation remains the only source of truth for W4A8 GMM2,
  packed-weight access, accumulator, D2/Fixpipe, and AIV dequant behavior.

Host ABI/cache repair completed:

- Rebuilt the CMake host extension target with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`:
  `cmake --build build/temp.linux-aarch64-cpython-312 --target vllm_ascend_C -- -j1`.
- Installed the rebuilt extension with:
  `cmake --install build/temp.linux-aarch64-cpython-312`.
- `strings vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` now contains the four-return schema ending
  with `Tensor gmm2_accumulator_int32`.
- Fresh Python registration now reports:
  `torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback(...) -> (Tensor gmm2_post_dequant, Tensor hidden_x_readback, Tensor hidden_scale_readback, Tensor gmm2_accumulator_int32)`.
- Four-NPU preflight passed: logical NPUs `0,1,2,3` are visible as `Ascend910B4`.

Evidence captured:

- Four-visible-NPU Python preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_visible_npu_python_pre_host_rebuild.log`
- Host extension rebuild log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_host_extension_vllm_ascend_C_rebuild.log`
- Host extension install log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_host_extension_cmake_install_after_rebuild.log`
- Installed extension schema-string audit:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_extension_schema_strings_after_install.log`
- Fresh torch-op schema audit:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_python_schema_after_host_rebuild.log`

Real-device Gate B/C probes after host ABI repair:

- Raw high-half D2 / int32 accumulator probe:
  - log:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_after_host_rebuild_true_top1_expert0.log`
  - summary:
    `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_int32_accumulator_after_host_rebuild_true_top1_expert0.json`
  - exit code: `1` because the raw D2 strict gate still fails.
  - key result: `official_gmm2_accumulator_int32_reference_passed: true`, exact mismatch count `0`,
    `actual_nonzero: true`.
  - raw high-half D2 result: finite/nonzero; strict reference failed with max abs `0.0009765625`, mean abs
    `4.444917431101203e-05`, FP16 ULP max `2`.
- Raw low-half D2 probe:
  - log:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_raw_c2_low_after_host_rebuild_true_top1_expert0.log`
  - summary:
    `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_low_after_host_rebuild_true_top1_expert0.json`
  - exit code: `1` because the raw D2 strict gate still fails.
  - key result: `official_gmm2_accumulator_int32_reference_passed: true`, exact mismatch count `0`,
    `actual_nonzero: true`.
  - raw low-half D2 result: finite/nonzero; strict reference failed with max abs `0.001953125`, mean abs
    `0.00014361337525770068`, FP16 ULP max `2`.
- Normal W4A8_DEBUG post-dequant probe:
  - log:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_post_dequant_after_host_rebuild_true_top1_expert0.log`
  - summary:
    `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_post_dequant_after_host_rebuild_true_top1_expert0.json`
  - exit code: `1` because strict post-dequant reference still fails.
  - key result: post-dequant active tensor is finite/nonzero, max abs `0.3963119685649872`, mean abs
    `0.05269600450992584`.
  - post-dequant reference error: max abs `0.00037679076194763184`, mean abs `1.82786079676589e-05`.
    Mean is under the predeclared `2e-05` threshold, but max exceeds the predeclared `2e-04` threshold.

Current interpretation:

1. The previous exit-139 blocker was an installed host-extension ABI/cache mismatch; it is fixed in the current
   local build/install.
2. The official GMM2 AIC accumulator producer is now proven for this diagnostic: accumulator readback is exact,
   nonzero, and uses the official packed W2 and Stage 2.1 packed hidden boundary.
3. The remaining Stage 2.2 failure is no longer an all-zero GMM2 output and no longer an AIC accumulator mismatch.
   It is downstream of the exact int32 accumulator, in the D2/Fixpipe rounding contract, raw half readback
   comparator, or AIV post-dequant reference path.
4. Do not relax tolerances to pass this gate. The next patch must be tied to the exact official Fixpipe/D2/AIV
   contract and should first determine whether the unfused reference is missing an official rounding/scale-bit
   detail or whether the debug readback is tapping the wrong D2 half/source point.
5. Stage 2.3+ and production `DispatchFFNCombineW4A8SVDQ` remain blocked/fail-closed.

## Stage 2.2 GMM2 Host Extension ABI Handoff - 2026-06-27T01:05Z

This section is the latest authoritative handoff before the environment rebuild. It supersedes the
`2026-06-27T00:40Z` install-probe section. The current blocker is an installed host-extension ABI/cache mismatch,
not a W4A8 numerical result.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Accepted prior Stage 2 evidence; unchanged. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | CANN debug-op metadata is six-output, but the loaded `vllm_ascend_C` torch extension still registers the older three-return schema. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B and Gate C. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints reaffirmed:

- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding.
- No public `torch_npu.npu_grouped_matmul` path was used, modified, reinterpreted, or debugged.
- No speculative scale formula, packed-weight repack, V2C/C2V lifecycle change, or production SVDQ path change was
  made.
- The active validation target remains the official W4A8 GMM2 Gate B accumulator/readback path, followed only then
  by Gate C post-dequant comparison.

New finding after the `00:40Z` crash:

- The repo-local aggregate CANN kernel config was regenerated and refreshed after the earlier crash:
  `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json`
  now declares the six expected outputs:
  `out`, `expert_token_nums`, `gmm2PostDequant`, `hiddenXReadback`, `hiddenScaleReadback`,
  `gmm2AccumulatorInt32`.
- That aggregate config has eight `SVDQW4A8GMM2DebugReadback_*` binary entries, and all referenced `.o` and `.json`
  files exist under the repo-local custom-op tree.
- The source tree already declares the four-return Python-facing debug op:
  `gmm2_post_dequant`, `hidden_x_readback`, `hidden_scale_readback`, `gmm2_accumulator_int32`.
- The loaded installed extension is stale. `torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback._schemas` still
  reports only three returned tensors:
  `gmm2_post_dequant`, `hidden_x_readback`, `hidden_scale_readback`.
- `strings vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` confirms the installed extension still
  contains the old schema string ending at `hidden_scale_readback`. It does not expose
  `gmm2_accumulator_int32`.

Evidence captured:

- Aggregate binary-info audit:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_aggregate_binary_info_after_fix.log`
- Python torch-op schema audit:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_python_schema_after_aggregate_fix.log`
- Installed extension schema-string audit:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_extension_schema_strings.log`
- Earlier exit-139 probe remains recorded at:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_true_top1_expert0.log`

Current interpretation:

1. Do not treat the `00:40Z` exit-139 probe as a Gate B accumulator result, a Gate C post-dequant result, or a
   W4A8 math failure. The host extension and CANN custom-op metadata were not ABI-consistent.
2. The next environment must rebuild/reinstall the Python extension so
   `vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` contains the four-return schema with
   `gmm2_accumulator_int32`.
3. After rebuild, first rerun only the schema audit with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` and confirm the torch
   schema returns four tensors.
4. Only after the torch schema and CANN aggregate metadata agree should the real-device Stage 2.2 probe be rerun.
   The first numerical gate remains `gmm2_accumulator_int32` Gate B; Gate C post-dequant remains downstream.

## Stage 2.2 GMM2 Int32 Accumulator Install Probe - 2026-06-27T00:40Z

This section is the latest authoritative handoff. It supersedes the `2026-06-27T00:13Z` rebuild handoff.
The six-output debug op now builds, installs into the repo-local custom-op tree, and registers from Python, but
the real-device Stage 2.2 probe currently crashes before producing any numerical Gate B/Gate C comparison. This
is not a Stage 2.2 numerical pass and it does not enable production SVDQ.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Accepted prior Stage 2 evidence; unchanged by the install refresh. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Six-output GMM2 debug op registers, but the real-device probe exits `139` before numerical comparison. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B and Gate C. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints reaffirmed:

- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding.
- No public `torch_npu.npu_grouped_matmul` path was used, modified, or debugged.
- No speculative scale formula, packed-weight repack, V2C/C2V lifecycle change, or production SVDQ path change was
  made.
- The active problem remains the official W4A8 GMM2 producer/consumer lifecycle and Gate B/Gate C validation.

Install and registration work completed:

- A broad `cmake --build csrc/build --target package -- -j1` attempt was started to refresh the `.run` package,
  but it entered a full selected-op binary rebuild and was interrupted with exit `130`. This is recorded only as
  packaging provenance; it is not numerical progress.
- `cmake --install csrc/build --prefix /tmp/svdq_stage2_gmm2_install` completed and staged the current custom-op
  artifacts.
- The repo-local install under `vllm_ascend/_cann_ops_custom/vendors/custom_transformer` was refreshed from the
  staged CMake install for:
  - `op_api/lib/libcust_opapi.so`
  - `op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json`
  - `op_api/include/aclnnop/aclnnInner_svdqw4_a8_gmm2_debug_readback.h`
  - `op_proto/inc/svdqw4_a8_gmm2_debug_readback_proto.h`
  - the GMM2 debug kernel directory and per-op kernel config.
- The generated per-op kernel dispatch config initially contained both stale five-output hashes and rebuilt
  six-output hashes. The stale five-output generated kernel artifacts were removed from the build output, then
  `csrc/cmake/scripts/util/ascendc_ops_config.py --skip-binary-info-config` regenerated
  `csrc/build/binary/ascend910b/bin/svdqw4_a8_gmm2_debug_readback.json`.
- The installed GMM2 debug kernel config now has exactly eight variants, and every variant exposes six outputs:
  `out`, `expert_token_nums`, `gmm2PostDequant`, `hiddenXReadback`, `hiddenScaleReadback`, and
  `gmm2AccumulatorInt32`.
- Python registration with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`,
  `ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer`,
  and the repo-local `libcust_opapi.so` passed:
  `torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback` is registered.

Real-device probe result:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, routed true top-1 expert 0, `--local-num-experts 8`, and
  `--swiglu-limit 451111`.
- NPU preflight showed four visible idle 910B4 devices.
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_true_top1_expert0.log`
- Exit code:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_true_top1_expert0.exitcode`
  contains `139`.
- No summary JSON was produced.
- The only runtime failure visible in the probe log is:
  `TBE Subprocess[task_distribute] raise error[], main process disappeared!`, followed by a shell
  `Segmentation fault (core dumped)` message.

Evidence files from this install/probe attempt:

- Interrupted broad package log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_package.log`
- Temporary CMake install log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_cmake_install_tmp.log`
- Clean installed config audit:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_installed_config_cleaned.txt`
- Registration log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_registration.log`
- NPU preflight:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260627T_stage2_gmm2_int32_accumulator_npu_smi.log`

Current interpretation and next work:

1. The previous binary-build blocker is fixed: the op-info, op API library, installed per-op kernel config, and
   installed debug kernel variants are six-output.
2. Stage 2.2 Gate B has not run to comparison; do not interpret the exit-139 probe as an accumulator mismatch or
   a zero/nonzero result.
3. After the environment rebuild, rerun a minimal device launch with the cleaned six-output install and inspect
   CANN/TBE crash logs before making behavioral kernel changes.
4. If the launch reaches Python output, validate `gmm2_accumulator_int32` first. Only after Gate B is finite,
   nonzero, and compared against the official-contract reference may Gate C post-dequant validation continue.

## Stage 2.2 GMM2 Int32 Accumulator Binary Rebuild - 2026-06-27T00:13Z

This section is the latest authoritative handoff. It supersedes the immediately following
`2026-06-26T23:45Z` rebuild handoff, whose source diagnosis was correct but whose binary-build blocker has now
been fixed. This is still not a Stage 2.2 numerical pass and it does not enable production SVDQ.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Accepted prior Stage 2 evidence; unchanged by this build-system patch. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | The debug op now builds with the six-output `gmm2AccumulatorInt32` ABI. Install/register and real-device Gate B accumulator validation are still pending. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B and Gate C. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding requirement update:

- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was reread in this turn and remains
  binding.
- No public `torch_npu.npu_grouped_matmul` path was used or modified.
- No W4A8 scale formula, packed-weight format, C2V/V2C lifecycle, or production SVDQ path was changed.
- The production `DispatchFFNCombineW4A8SVDQ` host tiling remains fail-closed.

Root cause corrected:

- The six-output op store and dynamic wrapper were regenerated, but the Ascend binary OPC compile rules could reuse
  stale `.done` files and stale `SVDQW4A8GMM2DebugReadback_*_param.json` inputs.
- `csrc/cmake/func.cmake` did not make each binary `.done` output depend on the generated compile script or copied
  dynamic Python, and the per-variant target did not depend on `generate_compile_cmd_${BINARY_COMPUTE_UNIT}`.
- As a result, OPC previously consumed five-output param JSON files even though the op store expected six outputs.

Files changed in this attempt:

- `csrc/cmake/func.cmake`
  - Added file-level dependencies from binary `.done` outputs to `${bin_script}` and `${DYNAMIC_PY_FILE}`.
  - Added a target-level dependency from each per-variant binary compile target to
    `generate_compile_cmd_${BINARY_COMPUTE_UNIT}`.

Validation commands and results:

- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target generate_compile_cmd_ascend910b -- -B -j1`:
  passed and regenerated the GMM2 debug compile scripts/param JSON files.
- Script/param audit:
  all eight `SVDQW4A8GMM2DebugReadback-svdqw4_a8_gmm2_debug_readback-*.sh` files now reference six-output param
  JSON files whose final output is `gmm2AccumulatorInt32`.
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`:
  passed. All eight six-output OPC variants generated successfully:
  `aa5b120529daaf84972f322c315db307`, `744db51fa6e1f42ff3e6b6a5fb044ace`,
  `8fd9c1b4c000e78db783d34628fa5afc`, `6158724529d46b254c9148d17de3caf6`,
  `9a44d5098210aa15a7c15cb9401632dc`, `45e122228f80275cada0ec3b998af3e8`,
  `326f3495d0f89f32f11d9af00c243c38`, and `b7fb8fe1ee499856e08a63895144ee73`.
- Saved build log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_int32_accumulator_build_fixed.log`
- Log grep result:
  no `invalid output nums`, no `Opc tool compile failed`, no `Traceback`, and no `CMake Error`.

Remaining work:

1. Install/register the rebuilt custom op package under `vllm_ascend/_cann_ops_custom`.
2. Run the real-device Stage 2.2 probe with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` and the installed six-output ABI.
3. Validate `gmm2_accumulator_int32` against the official-contract int32 accumulator reference.
4. Keep Stage 2.2 failed until Gate B and Gate C both pass; do not advance to SVDQ down composition or production
   host tiling before those gates pass.

## Stage 2.2 GMM2 Int32 Accumulator Readback ABI - 2026-06-26T23:45Z

This section is the latest rebuild handoff. It records the current source state after adding a debug-only GMM2
int32 accumulator readback path that reuses the official W4A8 GMM2 AIC producer. This is not a numerical gate
pass and it does not enable production SVDQ.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Accepted prior Stage 2 evidence; unchanged by this patch. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Source is wired for an official-producer int32 accumulator readback, but the Ascend binary build is blocked before a real-device probe because OPC still consumes stale 5-output `_param.json` files while the op store is 6-output. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B and Gate C. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Source changes in this attempt:

- Added a debug-only `CopyL0CToGm<int32_t -> int32_t, NO_QUANT>` tap inside the official W4A8 `BlockMmad` path.
- Threaded an optional `debugGMM2AccumulatorGM` pointer through `DispatchFFNCombineW4A8`, `MatmulKernel::Params`,
  the GMM2 AIC call site, the debug kernel ABI, ACLNN wrapper, torch adapter, torch schema, and meta registration.
- Added a fourth debug-op return tensor named `gmm2_accumulator_int32` with shape
  `[max_output_size * 2, hidden_size]`.
- Extended `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py` with a diagnostic-only int32 accumulator reference
  comparator. This host comparator is only for readback validation; it is not an alternate GMM2 execution path.
- Updated the ABI unit test to require `gmm2AccumulatorInt32` wiring.

Validation commands and results:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check`: passed.
- `pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered -q`: passed.
- `pytest -q tests/ut/ops/test_svdq_moe_abi.py -q`: failed in three pre-existing low-rank BF16 ABI tests unrelated to the GMM2 debug ABI:
  `test_svdq_lowrank_debug_mmad_synchronizes_l0_reuse`,
  `test_svdq_cann_lowrank_down_up_component_contract_is_wired`, and
  `test_svdq_kernel_contract_manifest_documents_workspace_sync_and_stage_map`.
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target generate_transformer_adapt_py -- -B -j1`: passed and regenerated the dynamic Python wrapper with the 6-output ABI.
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`: reported target success but OPC emitted errors for every generated binary variant, so this is treated as failed for the Stage 2.2 diagnostic gate.

Evidence logs:

- Initial build attempt:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_int32_accumulator_build.log`
- Regenerated wrapper / rerun build:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_int32_accumulator_build_rerun.log`
- Relevant OPC failure from the rerun:
  `Op[type=SVDQW4A8GMM2DebugReadback] invalid output nums[5], which should be equal to output nums[6] in op store.`

Current diagnosis:

- The regenerated op store, ACLNN inner autogen, proto, op-info JSON, and dynamic Python wrapper all include
  `gmm2AccumulatorInt32`.
- The OPC input files under `csrc/build/binary/ascend910b/gen/SVDQW4A8GMM2DebugReadback_*_param.json` still list
  only five outputs through `hiddenScaleReadback`.
- Because the binary `_param.json` files remain stale/incomplete, no real-device accumulator probe was run and no
  Gate B or Gate C progress is claimed.

Next rebuild action:

1. Regenerate or invalidate the Ascend binary param JSON/scripts for `SVDQW4A8GMM2DebugReadback` so the OPC
   input param files include `gmm2AccumulatorInt32` as output index 5.
2. Rebuild `svdqw4_a8_gmm2_debug_readback_ascend910b` with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` and require zero
   `invalid output nums` / `Opc tool compile failed` lines in the saved log.
3. Only after the binary build is clean, install/register the updated custom op and run the real-device Stage 2.2
   accumulator probe. Do not run public `torch_npu.npu_grouped_matmul`; do not enable production host tiling.

## Stage 2.2 Official-Path Appendix Rebaseline - 2026-06-26T22:52:59Z

This section is the latest authoritative Stage 2.2 handoff. It records that
`svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was reread and is binding for
all remaining Stage 2.2 work. Older sections below are historical evidence unless explicitly referenced here.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | The latest Gate A manifest rerun proves the modified-hidden input boundary is deterministic, finite/nonzero, packed correctly, and post-override readback exact for the true top-1 expert-0 probe. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A is explicit in the summary JSON; Gate B raw D2 high-half is finite/nonzero but still fails the strict comparator (`max_abs: 0.0009765625`, `mean_abs: 0.00004444917431101203`, failed elements `2227`). |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B and Gate C. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints confirmed from the appendix:

- Do not modify, reinterpret, or debug the public `torch_npu.npu_grouped_matmul` path.
- Do not reopen the solved BF16 producer work or Stage 2.1 packed-hidden work.
- Use the official `dispatch_ffn_combine_w4_a8` lifecycle as the only behavioral source of truth for GMM2
  AIC, C2V handoff, `BlockEpilogue2`, `CombineV2`, packed W2 access, tiling, accumulator/Fixpipe, and AIV
  dequantization.
- Do not make speculative changes to V2C/C2V flags, token state, cumsum state, core ownership, workspace
  offsets, or synchronization unless the exact official counterpart is cited in the state table.
- Keep Stage 2.3+, SVDQ down composition, final combine, and production host tiling blocked until Stage 2.2
  Gate B and Gate C pass on real Ascend hardware.

Current accepted evidence:

- Gate A manifest summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_gate_a_manifest_high_half_true_top1_expert0.json`
- Real-device probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_gate_a_manifest_high_half_true_top1_expert0.log`
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_gate_a_manifest_npu_smi.log`
- The probe used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` and exited `1` as expected because the strict Gate B
  comparator remains failed.

Next permitted work:

1. Preserve the full official lifecycle path where `gmm2OnlyFromPacked_` remains false and only the GMM2 hidden
   packed input plus hidden-scale boundary is overridden.
2. If adding a new diagnostic, expose a real official Gate B boundary only: accumulator, Fixpipe output, D2
   source region, or C2V-fed `BlockEpilogue2` input. The tap must reuse the official AIC producer and official
   lifecycle; it must not introduce an alternative GEMM or host-side substitute path.
3. Record the exact official-vs-debug state-table deviation corrected before any behavioral patch, then run the
   device probe with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.

Files changed in this attempt:

- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Validation:

- This is a report-only rebaseline; no source behavior changed and no numerical gate progress is claimed.

## Stage 2.2 Gate A Manifest Plumbing and Rerun - 2026-06-26T22:43:36Z

This section is the latest Stage 2.2 status for the environment rebuild. It adds explicit Gate A input-boundary
metadata to the existing modified-hidden GMM2 probe and records one real-device rerun.

The local UTC clock for this update is `2026-06-26T22:43:36Z`. The immediately following state-table section
has a later manual timestamp in its heading, but it is historical relative to this top section.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Gate A manifest rerun confirms exact packed-hidden override readback. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A manifest is now emitted; Gate B high-half comparator still fails strict tolerance. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Extended tensor byte manifests with device, storage size, logical byte size, element size, and contiguity.
  - Added sampled metadata manifests for large W2 tensors so the real checkpoint weight metadata is recorded without
    copying or hashing full packed weights.
  - Added `gate_a_input_boundary` to every Stage 2.2 probe summary mode.
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
  - Added this status section.

New `gate_a_input_boundary` fields include:

- `canonical_hidden_bf16`
- `hidden_int8`
- `hidden_int4_packed_active`
- `hidden_int4_packed_full_padded`
- `hidden_scale_active`
- `hidden_scale_full_padded`
- `expert_token_nums`
- active expert IDs, expert prefix sums, expert-local row starts, expert-local row offsets, and first row map entries
- active-row count and padded-row interpretation
- sampled W2 packed weight metadata, W2 scale metadata, and W2 scale-bias metadata
- official source contract notes for packed hidden workspace, hidden-scale workspace, and W2 access

Validation commands:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- NPU preflight:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 npu-smi info`
  - Log:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_gate_a_manifest_npu_smi.log`
- Real-device probe command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --swiglu-limit 451111 --summary-name phase_stage2_gmm2_gate_a_manifest_high_half_true_top1_expert0.json`
  - Log:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_gate_a_manifest_high_half_true_top1_expert0.log`
  - Exit code file:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_gate_a_manifest_high_half_true_top1_expert0.exitcode`
  - Summary:
    `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_gate_a_manifest_high_half_true_top1_expert0.json`

Rerun result:

- Probe exit code: `1`, expected because strict Stage 2.2 Gate B remains failed.
- Summary stage: `stage2_modified_hidden_official_w4a8_gmm2_raw_c2`.
- `gate_a_input_boundary` present: `true`.
- Gate A manifest status: `diagnostic_manifest_only`.
- Active rows: `16`.
- `expert_token_total_matches_active_rows`: `true`.
- Padded rows: hidden INT4 zero `true`, hidden scale zero `true`.
- `canonical_hidden_bf16.dtype`: `torch.bfloat16`.
- `hidden_int8.dtype`: `torch.int8`.
- W2 packed metadata shape: `[8, 512, 256]`.
- W2 scale metadata shape: `[8, 1, 2048]`.
- `hidden_post_override_readback_exact`: `true`.
- `hidden_scale_post_override_readback_exact`: `true`.
- `official_gmm2_aic_raw_output_finite`: `true`.
- `official_gmm2_aic_raw_output_nonzero`: `true`.
- `official_gmm2_aic_reference_passed`: `false`.
- `official_gmm2_numerical_gate_passed`: `false`.
- Raw high-half comparator remains failed:
  - `max_abs: 0.0009765625`
  - `mean_abs: 0.00004444917431101203`
  - failed elements over strict tolerance: `2227`

Conclusion:

- Gate A evidence is now emitted in a single explicit summary section and was verified in a real-device run.
- This does not close Stage 2.2. The remaining active boundary is still Gate B: exact official GMM2 D2/Fixpipe
  high/low-half semantics before post-dequant/final output.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

## Stage 2.2 Official-vs-Debug GMM2 State Table - 2026-06-26T23:05Z

This section is the latest Stage 2.2 status for the environment rebuild. It responds to
`svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`: before another behavioral
patch, the official GMM2 lifecycle and the current isolated debug behavior must be recorded side by side.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Current Stage 2.2 probes still use the validated packed hidden INT4 and hidden-scale boundary. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Required official-vs-debug lifecycle table is now recorded; no new numerical gate is claimed. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Important interpretation:

- The current debug entry point is named `InitGMM2OnlyFromPacked`, but it deliberately leaves
  `gmm2OnlyFromPacked_ = false`
  (`csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:214-228`).
- That means the debug operator runs the full official `GMM1 -> GMM2 -> DispatchAndCombine` AIC/AIV lifecycle
  (`dispatch_ffn_combine_w4_a8_kernel.hpp:250-272`) and overrides only the GMM2 hidden packed-input and scale
  boundary inside the official AIV path (`dispatch_ffn_combine_w4_a8_kernel.hpp:1355-1368`).
- The unused `params.gmm2OnlyFromPacked == true` path seeds `tokenPerExpert`, `cumsumMM`, `preSumBeforeRank`, and
  hidden buffers directly (`dispatch_ffn_combine_w4_a8_kernel.hpp:913-1007`). That path is a useful source
  reference, but it is not the currently preferred official-lifecycle debug path.
- This report update is not Gate A/B/C progress. It is the mandatory state table required before the next
  behavior change.

Official source locations inspected:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/svdqw4_a8_gmm2_debug_readback.cpp:14-31`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:214-338`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:250-315`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:684-785`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:913-1007`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1295-1403`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1439-1518`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1520-1650`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp:174-424`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:100-310`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:380-470`
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_gmm2_debug_readback.cpp:14-59`
- `csrc/mc2/svdq_w4a8_gmm2_debug_readback/svdq_w4a8_gmm2_debug_readback_torch_adpt.h:42-150`

Official-vs-debug state table:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` / `gmA2I4` | `BlockEpilogue1` high/low INT4 pack writes `gmA2I4_I8` from GMM1/SwiGLU hidden (`block_epilogue_w4a8post_pertoken_swiglu.hpp:351-424`). | GMM2 AIC consumes `gmA2I4` in `blockMmad(... gmA2I4[gmGroupOffsetA + gmOffsetA] ...)` (`dispatch_ffn_combine_w4_a8_kernel.hpp:765-767`). | `initBuffer` binds both typed views to `workspaceInfo.ptrA2Int4` (`dispatch_ffn_combine_w4_a8_kernel.hpp:290-294`). | `ptrA2Int4 = ptrWorkspace + offset_after ptrA1Int4`; size advances by `maxOutputSize * k2` bytes (`dispatch_ffn_combine_w4_a8_kernel.hpp:1588-1593`). | GMM2 uses `layoutA2.GetTileLayout(M,K)` with `k2 = problemShape.n()/2`; row-major INT4 view advances by `inGroupProblemShape.m() * inGroupProblemShape.k()` (`dispatch_ffn_combine_w4_a8_kernel.hpp:690-735`, `772-773`). | C2V/V2C lifecycle uses `SYNCFLAGC2V` then `SYNCFLAGV2C`; BlockEpilogue1 uses MTE/V events for pack writes. | Official AIV sets `SYNCFLAGV2C` after each SwiGLU/hidden-pack chunk (`dispatch_ffn_combine_w4_a8_kernel.hpp:1371-1373`). | GMM2 waits `SYNCFLAGV2C` at group 0 or sync-group boundaries (`dispatch_ffn_combine_w4_a8_kernel.hpp:742-744`). | `blockEpilogue1.Finalize()` drains pack events before `CombineV2` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1375`, `1403`). | Full official path runs, then current debug overrides only the same `gmA2I4_I8` slice from `externalHiddenX` before signaling GMM2 (`dispatch_ffn_combine_w4_a8_kernel.hpp:1355-1368`). |
| hidden scale `gmPerTokenScale2` | `BlockEpilogue1` writes per-token hidden scale after hidden quantization (`block_epilogue_w4a8post_pertoken_swiglu.hpp:362-365`, `418-423`). | `BlockEpilogue2` multiplies FP32 D2 rows by `gmPerTokenScale2` (`block_epilogue_w4a8post_pertoken_v2.hpp:300-307` equivalent for first epilogue; V2 uses `gmPerTokenScale` in its post-dequant scale loop). | `initBuffer` binds `gmPerTokenScale2` to `workspaceInfo.ptrPerTokenScale2` (`dispatch_ffn_combine_w4_a8_kernel.hpp:304-307`). | `ptrPerTokenScale2 = ptrWorkspace + offset_after ptrPerTokenScale`; size `maxOutputSize * sizeof(float)` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1558-1563`). | Vector layout, one FP32 scale per logical active hidden row. | Same chunk-level `SYNCFLAGV2C` protects scale visibility to GMM2/CombineV2. | Official AIV writes scale before `SYNCFLAGV2C`. | GMM2 waits `SYNCFLAGV2C`; CombineV2 waits GMM2 completion flags before reading D2 and scale. | BlockEpilogue2 finalizes UB/MTE3 events (`block_epilogue_w4a8post_pertoken_v2.hpp:130-140`). | Current debug copies `externalHiddenScale` into the same `gmPerTokenScale2[rowStartThisCore]` slice before `SYNCFLAGV2C` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1363-1367`). |
| `tokenPerExpert` | Official routing/cross-rank path populates peer-memory `tokenPerExpert`; `initBuffer` binds it through `shmem() + offsetPeerTokenPerExpert` (`dispatch_ffn_combine_w4_a8_kernel.hpp:309-313`). | GMM1/GMM2 cumsum derivation and `BlockEpilogue2` remote write routing consume it (`block_epilogue_w4a8post_pertoken_v2.hpp:114-115`, `278-279`). | `initBuffer` and later official routing/peer all-gather initialize the region. | Peer memory at `peermemInfo.offsetPeerTokenPerExpert = shmem.SegmentSize() - 2 * MB_SIZE` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1647-1650`). | `Layout3D(AlignUp(EP * expertPerRank + 1, 128), expertPerRank)` (`dispatch_ffn_combine_w4_a8_kernel.hpp:312-313`). | Cross-rank peer-memory waits use `gm_signal_wait_until_ne`; GMM flags derive from cumsum state. | Official routing/all-gather writes before GMM1/GMM2. | GMM1 waits cumsum-ready flag; GMM2 waits V2C chunk flags. | Official path resets token state after combine (`dispatch_ffn_combine_w4_a8_kernel.hpp:1404-1405`). | Current full-lifecycle debug uses official token state, not direct seeding. The inactive direct-seed helper would copy `externalExpertTokenNums` to token state (`dispatch_ffn_combine_w4_a8_kernel.hpp:935-950`). |
| `cumsumMM` | Official AIV computes cumulative expert counts via `GetCumsumForMMAIV` or routing state. | GMM1, GMM2, BlockEpilogue1, and CombineV2 all use `cumsumMM((EP-1)*expertPerRank + groupIdx)` for group row counts. | `initBuffer` binds `cumsumMM` to `workspaceInfo.ptrcumsumMM` (`dispatch_ffn_combine_w4_a8_kernel.hpp:284`). | `ptrcumsumMM = ptrWorkspace + AlignUp(M,256)*topK*sizeof(int32_t)`; size then advances by `EP*EP*expertPerRank*sizeof(int32_t)` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1550-1555`). | Per-EP by per-expert cumulative INT32 table. | CrossCore cumsum-ready and V2C/C2V flags. | Official cumsum is ready before GMM1/GMM2 loops read it. | GMM2 reads before scheduling each expert group (`dispatch_ffn_combine_w4_a8_kernel.hpp:710-719`). | No independent drain; consumed through GMM and epilogue lifecycle. | Current full-lifecycle debug uses official cumsum. Inactive direct seed would derive it from `externalExpertTokenNums` (`dispatch_ffn_combine_w4_a8_kernel.hpp:945-949`). |
| `preSumBeforeRank` | Official cross-rank token all-gather computes per-rank prior row sums (`dispatch_ffn_combine_w4_a8_kernel.hpp:1010-1131`). | `BlockEpilogue2` uses it to compute destination offsets for remote peer output (`block_epilogue_w4a8post_pertoken_v2.hpp:278-306`). | `initBuffer` binds `preSumBeforeRank` to `workspaceInfo.ptrSumBeforeRank` (`dispatch_ffn_combine_w4_a8_kernel.hpp:314`). | `ptrSumBeforeRank` follows optional debug `ptrCGMM2`; size `EP * expertPerRank * sizeof(int32_t)` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1611-1621`). | `[EP, expertPerRank]` INT32 table. | Peer all-gather waits/checks; no standalone flag in CombineV2. | Official setup completes before `DispatchAndCombine` reaches `CombineV2`. | `BlockEpilogue2` reads during remote output copy. | No separate drain; remote copy drains through epilogue events. | Current full-lifecycle debug uses official `preSumBeforeRank`. Inactive direct-seed path zeros it (`dispatch_ffn_combine_w4_a8_kernel.hpp:923-931`, `950`), which is valid only for single-rank style debugging. |
| GMM2 AIC input tile state | Official GMM2 constructs `inGroupProblemShape{currentM*2, n2, k2}`, `layoutA2`, `layoutB2`, `layoutScale2`, and `layoutC` per expert (`dispatch_ffn_combine_w4_a8_kernel.hpp:710-737`). | `BlockMmad` consumes A2 INT4, W2 INT4, W2 scale, and writes D2/C2. | `Process` defines official W4A8 `BlockMmad` policy and layouts (`dispatch_ffn_combine_w4_a8.h:256-307`). | A2 is `ptrA2Int4`; B2 and scale2 are postloaded tensor-list addresses selected through `GetTensorAddr` (`dispatch_ffn_combine_w4_a8_kernel.hpp:722-727`). | L1 tile `[128,256,1024]`, L0 tile `[128,256,256]`; GMM2 N=`hidden`, K=`intermediate/2` (`dispatch_ffn_combine_w4_a8.h:259-285`, `dispatch_ffn_combine_w4_a8_kernel.hpp:690-735`). | Waits `SYNCFLAGV2C`; later `blockMmad.Finalize` sets completion flags. | AIV signals after hidden pack/override. | AIC waits before group scheduling. | `blockMmad.SynchronizeBlock(); blockMmad.Finalize(expertPerRank-1, 0)` (`dispatch_ffn_combine_w4_a8_kernel.hpp:781-783`). | Current debug uses this exact GMM2 AIC loop and packed W2 access. |
| GMM2 accumulator / D2 region | Official `BlockMmad` writes `gmC2[gmGroupOffsetC + gmOffsetC]` (`dispatch_ffn_combine_w4_a8_kernel.hpp:765-767`). | `CombineV2` / `BlockEpilogue2` reads `gmC2` as high/low D2 halves (`block_epilogue_w4a8post_pertoken_v2.hpp:150-172`). | `initBuffer` binds `gmC2` to `workspaceInfo.ptrC2` (`dispatch_ffn_combine_w4_a8_kernel.hpp:301-302`). | `ptrC2` follows `ptrC`; W4A8 size advances by `maxOutputSize * n2 * sizeof(ElementC) * 2` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1574-1580`). | D2 stores doubled high/low rows; V2 epilogue reinterprets with `actualBlockShape.m()/2` logical rows and `n2` columns. | GMM2 `Finalize` completion flags consumed by `CombineV2`. | AIC finalizes after outstanding blocks drain. | CombineV2 waits one flag per group before BlockEpilogue2 (`dispatch_ffn_combine_w4_a8_kernel.hpp:1490-1493`). | `BlockEpilogue2.Finalize()` drains UB/MTE events (`block_epilogue_w4a8post_pertoken_v2.hpp:130-140`). | Current debug reads the official D2 region through `BlockEpilogue2`; failed pre-Fixpipe accumulator tap was reverted because no supported no-quant L0C copy specialization exists. |
| C2V handoff state | Official GMM2 `blockMmad.Finalize(..., 0)` publishes completion for AIV CombineV2 (`dispatch_ffn_combine_w4_a8_kernel.hpp:781-783`). | `CombineV2` waits completion flags before each group tile (`dispatch_ffn_combine_w4_a8_kernel.hpp:1490-1493`). | The flag protocol is owned by `BlockMmad`/CATLASS. | CrossCore flag space, not GM tensor payload. | Per sync group / expert group. | `CrossCoreWaitFlag<0x2>(flag_id)` in AIV. | AIC finalizes after all block work. | AIV waits before `blockEpilogue(...)`. | Final wait loop consumes remaining flags (`dispatch_ffn_combine_w4_a8_kernel.hpp:1512-1515`). | Current debug keeps this official C2V handoff because `gmm2OnlyFromPacked_` remains false. |
| `BlockEpilogue2` input state | Official constructor receives EP, expertPerRank, rank, peer token pointer, layoutD2, n2, n0, shmem, offsetD, raw-debug mode (`dispatch_ffn_combine_w4_a8_kernel.hpp:1316-1327`). | It reads `gmC2`, aux bias `ptrMAux2`, `gmPerTokenScale2`, `tokenPerExpert`, and `preSumBeforeRank`, then writes peer output. | Constructed in `DispatchAndCombine` before hidden pack loop; `InitFlag()` is called inside `CombineV2` (`dispatch_ffn_combine_w4_a8_kernel.hpp:1329-1331`, `1439-1442`). | Peer output uses `shmem(params.offsetD, dstEpIdx)` (`block_epilogue_w4a8post_pertoken_v2.hpp:292-306`). | 32-row AIV subtiles over `actualBlockShape.m()/2`; column tile `n0`. | UB stage MTE2/V/MTE3 events. | After GMM2 completion waits in CombineV2. | `InitFlag` primes UB copy events; per-tile waits guard reads/writes. | `Finalize` waits all UB stage events. | Current debug uses the same `BlockEpilogue2`; raw debug sentinel modes only redirect the optional FP32 tap. |
| FP32 post-dequant debug tap | Official production path does not need this tensor; W4A8 debug mode sets `ptrCGMM2` to `params.ptrDebugGMM2` when provided (`dispatch_ffn_combine_w4_a8_kernel.hpp:1611-1618`). | Python probe reads the returned `gmm2_post_dequant` tensor. | Torch adapter allocates `{maxOutputSize, hidden_size}` FP32 output (`svdq_w4a8_gmm2_debug_readback_torch_adpt.h:119-124`). | `ptrCGMM2` is either workspace or debug output GM; debug output is passed as `gmm2PostDequant` from ACLNN. | V2 epilogue debug offset `gmCOffset = (preSrcExpertSum*n2 + blockCoord.m()*n2)/2 + blockCoord.n()` (`block_epilogue_w4a8post_pertoken_v2.hpp:173-176`). | `EVENT_ID7` protects debug GM writes. | V2 epilogue writes after high/low combine and after optional aux/scale depending on raw mode. | Probe reads after op completion. | `EVENT_ID7` is finalized in `BlockEpilogue2::Finalize()` (`block_epilogue_w4a8post_pertoken_v2.hpp:130-140`). | Current debug tap is official AIV post-dequant/optional raw D2, not a pre-Fixpipe accumulator. The attempted pre-Fixpipe tap failed at compile time and remains uncommitted. |

Deviations that are now explicit:

- The only intended current behavioral deviation from official production is the AIV hidden-boundary override:
  `gmA2I4_I8` and `gmPerTokenScale2` are replaced from validated external Stage 2.1 tensors immediately before
  the official `SYNCFLAGV2C` handoff.
- The currently executed path keeps official GMM2 AIC, official C2V handoff, official `BlockEpilogue2`, and official
  final drain. Switching `gmm2OnlyFromPacked_` to `true` would move to a more isolated but less official state
  reconstruction path and is therefore not the preferred next correction.
- Remaining evidence gaps from the appendix are Gate A row-identity/prefix/padded-row proof and Gate B exact
  accumulator/Fixpipe-boundary proof. Existing D2 half readbacks are finite and nonzero but still fail strict
  numerical matching, so Stage 2.2 remains open.

Files changed in this attempt:

- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Validation:

- No source behavior changed.
- `git diff --check -- docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`: passed.

## Stage 2.2 Pre-Fixpipe Accumulator Tap Attempt - 2026-06-26T22:17:48Z

This section is the latest Stage 2.2 status for the environment rebuild. It records an attempted
debug-only route to the official GMM2 pre-Fixpipe accumulator boundary and the reason that route was
not kept in source.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Current Stage 2.2 probes still use the validated packed hidden INT4 and hidden-scale boundary. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | The direct no-quant pre-Fixpipe accumulator tap does not compile with the available CATLASS AtlasA2 copy path. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Attempted diagnostic:

- Added a temporary debug-only sentinel range `460000 < swiglu_limit < 462000`.
- Intended to copy the full GMM2 `int32_t` L0C accumulator into the existing FP32 GMM2 debug output before
  `SetFixPipeConfig<uint64_t, false>`, `VDEQF16`, FP16 storage, aux bias, hidden-scale multiplication, or
  peer-output routing.
- Intended row layout was high-accumulator and low-accumulator rows interleaved for each logical hidden row.
- This was only an isolated debug experiment; it did not alter production `DispatchFFNCombineW4A8SVDQ`.

Build command:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1
```

Evidence:

- Build log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_prefix_accumulator_debug_build.log`
- The relevant compile failure is:
  `Unsupported copy l0c to gm, can not find the specialization.`
- The missing specialization is:
  `CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, int, float, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>`.
- The failing instantiation came from the temporary `BlockMmad` accumulator tap using
  `CopyL0CToGm<AtlasA2, int32_t, GemmType<float, RowMajor>, NO_QUANT, false>`.

Conclusion:

- Direct `int32_t` L0C accumulator to FP32 GM copy through the existing CATLASS no-quant `CopyL0CToGm`
  path is unsupported on this AtlasA2 specialization.
- The unsupported source changes were reverted and are not committed.
- This attempt is negative evidence only. It is not progress toward the Stage 2.2 numerical gate.
- The next valid accumulator-boundary route must use a supported official/device path, for example a
  supported int32 GM debug output mode, a dedicated AscendC accumulator extraction path if available, or a
  separately proven reproduction of the official hardware `VDEQF16` semantics.
- Do not relax tolerances, switch to public `torch_npu.npu_grouped_matmul`, or enable production host tiling
  on this evidence.

Validation after reverting the unsupported tap:

- Source files touched by the temporary tap were restored to the previously committed Stage 2.2 state.
- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check -- docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`: passed.

## Stage 2.2 Official Fixpipe Contract Handoff

This section is the latest Stage 2.2 status for the environment rebuild. Older sections are historical
evidence unless explicitly referenced here.

Logged during the current session with local UTC clock reading `2026-06-26T21:29:08Z`. The immediately
following `2026-06-26T21:35Z` section is retained as historical evidence from the prior handoff context.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Current Stage 2.2 top-1 runs still use the validated packed hidden INT4 and hidden-scale boundary. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | D2 high/low halves are finite and nonzero but still fail strict Gate B. Official Fixpipe contract has now been recorded for the next comparator boundary. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

This handoff update does not claim numerical gate progress. It records the concrete official
accumulator-to-Fixpipe contract that must be matched next, and keeps the failed public
`torch_npu.npu_grouped_matmul` path out of scope.

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added `official_fixpipe_contract` under the Stage 2.2 raw-C2 `gmm2` summary.
  - The helper is diagnostic/reporting-only and does not change raw-C2 references, strict tolerances, or
    pass/fail behavior.
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
  - Added this rebuild handoff section.

Official GMM2 Fixpipe contract now recorded in probe JSON:

- Accumulator type: `AscendC::int4b_t x AscendC::int4b_t -> int32_t`
  (`csrc/third_party/catlass/include/catlass/gemm/helper.hpp:137-140`).
- Tile-copy selection: `QuantTileCopy` chooses `CopyL0CToGm<AtlasA2, ElementAccumulator, CType,
  ScaleGranularity::PER_CHANNEL, false>` and copies scale as a `uint64_t` vector through GM -> A1 ->
  Fixpipe scale buffer (`csrc/third_party/catlass/include/catlass/gemm/tile/tile_copy.hpp:218-230`).
- Quant mode: `int32_t -> half` with `ScaleGranularity::PER_CHANNEL` selects
  `QuantMode_t::VDEQF16`
  (`csrc/third_party/catlass/include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp:144-150`).
- Row-major per-channel Fixpipe specialization:
  `AscendC::FixpipeParamsV220` sets `nSize = dstLayout.shape(1)`,
  `mSize = dstLayout.shape(0)`, `srcStride = srcLayout.stride(3) / srcLayout.stride(0)`,
  `dstStride = dstLayout.stride(0)`, `quantPre = VDEQF16`, `reluEn = false`, and forwarded
  `unitFlag`; then calls `AscendC::SetFixPipeConfig<uint64_t, false>(scale, false)`,
  `AscendC::PipeBarrier<PIPE_FIX>()`, and
  `AscendC::Fixpipe<half, int32_t, AscendC::CFG_ROW_MAJOR>(...)`
  (`copy_l0c_to_gm.hpp:344-364`).
- This specialization does not explicitly set `isChannelSplit`; do not assume that flag explains the
  current residuals.

Current numerical evidence carried forward:

- High D2 half readback (`--swiglu-limit 451111`):
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_high_half_rounding_variants_top1_expert0_max64.json`
  - Strict Gate B failed: `max_abs: 0.0009765625`, `mean_abs: 0.00004492711741477251`,
    failed elements `9010`.
  - Best tested mean-error variant remains `scale_low32_fp16_toward_zero`, but it still fails.
- Low D2 half readback (`--swiglu-limit 453333`):
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_low_half_rounding_variants_top1_expert0_max64.json`
  - Strict Gate B failed: `max_abs: 0.001953125`, `mean_abs: 0.00014362166984938085`,
    failed elements `38381`.
  - Best tested variant remains `scale_low32_fp16_toward_zero`, but it still fails.

Next valid work after rebuild:

1. Rebuild/install the current custom op surface and verify
   `torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback` registration.
2. Re-run one raw-C2 D2 half probe and confirm the new JSON contains
   `gmm2.official_fixpipe_contract`.
3. Continue below the current D2 half host approximation boundary by exposing or proving the official
   hardware `VDEQF16` accumulator-to-FP16 behavior, without changing the official GMM2 lifecycle.
4. Do not enable production `DispatchFFNCombineW4A8SVDQ` host tiling until the isolated real-device
   Stage 2.2 numerical gates pass.

Post-handoff real-device rerun:

- Command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --swiglu-limit 451111 --summary-name phase_stage2_gmm2_fixpipe_contract_high_half_top1_expert0_max64.json`
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_fixpipe_contract_high_half_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_fixpipe_contract_high_half_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_fixpipe_contract_high_half_top1_expert0_max64.json`
- Probe exit: `1`, expected because strict Gate B still fails.
- `torch_op_registered: true`.
- `stage.gmm2.official_fixpipe_contract` is present.
- Recorded contract fields:
  - `copy_l0c_to_gm_quant_pre: QuantMode_t::VDEQF16`
  - `is_channel_split_explicitly_set: false`
- Gate B high-half comparator remains failed:
  - `max_abs: 0.0009765625`
  - `mean_abs: 0.00005377935303840786`
  - failed elements over strict abs tolerance: `12621`
  - NaN/Inf counts: `0`
- Carry-forward checks from the summary:
  - `official_gmm2_entry_reached: true`
  - `official_gmm2_aic_raw_output_finite: true`
  - `official_gmm2_aic_raw_output_nonzero: true`
  - `official_gmm2_c2v_handoff_verified: true`
  - `hidden_packed_exact: true`
  - `hidden_post_override_readback_exact: true`
  - `hidden_scale_post_override_readback_exact: true`

True top-k=1 FP16 ULP diagnostic:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py` now adds diagnostic-only FP16 bit/ULP fields to the
  strict raw-C2 D2 half comparator, hidden-readback comparator, and D2 half rounding/scale variants.
  Gate logic and tolerances are unchanged.
- The previous `phase_stage2_gmm2_fixpipe_contract_high_half_top1_expert0_max64.json` filename was
  misleading: the summary proves it used default `top_k: 8` and routed experts `[0,1,2,3,4,5,6,7]`.
  New evidence below uses real `--top-k 1 --route-experts 0 --local-num-experts 8`.
- High half command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --swiglu-limit 451111 --summary-name phase_stage2_gmm2_d2_high_half_fp16_ulp_true_top1_expert0.json`
  - NPU preflight:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_high_half_fp16_ulp_true_top1_expert0_npu_smi.log`
  - Probe log:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_high_half_fp16_ulp_true_top1_expert0.log`
  - Summary:
    `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_high_half_fp16_ulp_true_top1_expert0.json`
  - Strict comparator: `max_abs: 0.0009765625`, `mean_abs: 0.00004444917431101203`,
    failed elements `2227`, FP16 exact ratio `0.531585693359375`, max ULP `2`, ULP > 1 count `259`.
  - Best variant: `scale_low32_fp16_toward_zero`, key `[0.00048828125, 0.000022813444957137108, 1136]`,
    FP16 exact ratio `0.7774658203125`, max ULP `2`, ULP > 1 count `6`.
- Low half command:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --top-k 1 --route-experts 0 --local-num-experts 8 --swiglu-limit 453333 --summary-name phase_stage2_gmm2_d2_low_half_fp16_ulp_true_top1_expert0.json`
  - NPU preflight:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_low_half_fp16_ulp_true_top1_expert0_npu_smi.log`
  - Probe log:
    `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_low_half_fp16_ulp_true_top1_expert0.log`
  - Summary:
    `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_low_half_fp16_ulp_true_top1_expert0.json`
  - Strict comparator: `max_abs: 0.001953125`, `mean_abs: 0.00014361337525770068`,
    failed elements `9569`, FP16 exact ratio `0.531768798828125`, max ULP `2`, ULP > 1 count `288`.
  - Best variant: `scale_low32_fp16_toward_zero`, key `[0.001953125, 0.00007748615462332964, 5221]`,
    FP16 exact ratio `0.755615234375`, max ULP `2`, ULP > 1 count `7`.

Conclusion from this diagnostic:

- The true single-expert case still fails, so the failure is not caused by the earlier top-k=8 grouping.
- Low32 scale plus toward-zero FP16 conversion is the closest tested host approximation for both D2 halves,
  but it is not exact and does not pass strict Gate B.
- The next useful boundary is below host rounding-mode substitution: expose/prove the official L0C
  accumulator before `SetFixPipeConfig<uint64_t, false>` / `VDEQF16`, or reproduce the exact hardware
  vector dequantization semantics in a dedicated isolated diagnostic. Do not relax tolerance or enable
  production on this evidence.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`: passed.
- Post-handoff high-half real-device rerun completed and wrote the summary listed above; exit `1` is expected
  because the strict comparator remains failed.
- `git diff --check -- docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`: passed after adding the
  post-handoff rerun evidence.
- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed after adding FP16 ULP
  diagnostics.
- `git diff --check -- tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed after adding FP16 ULP
  diagnostics.
- True top-k=1 high-half and low-half ULP real-device probes completed and wrote the summaries listed above;
  both exited `1` as expected because strict Gate B remains failed.

## Stage 2.2 D2 Half Rounding/Scale Variant Diagnostic - 2026-06-26T21:35Z

This section is historical evidence. Older sections are historical evidence unless explicitly
referenced here.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Current Stage 2.2 top-1 runs still use the validated packed hidden INT4 and hidden-scale boundary. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | D2 high/low halves remain finite and nonzero but fail strict Gate B. New variant diagnostics reject high32 scale-word interpretation and common FP16 rounding modes as sufficient explanations. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

This attempt is host-probe/report work only. It does not modify device kernel code, synchronization, GMM2
lifecycle, packed-weight access, public grouped matmul usage, tolerance values, or production SVDQ host
tiling. It adds diagnostic-only comparisons against narrow Fixpipe candidate contracts while keeping the
strict Gate B pass/fail comparator unchanged.

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added D2 half variant diagnostics for:
    - W2 scale low 32-bit FP32 interpretation;
    - W2 scale high 32-bit FP32 interpretation;
    - FP16 nearest-even, toward-zero, floor, ceil, and no-FP16-rounding candidates.
  - The existing `raw_c2_unfused_reference.passed` gate remains unchanged.

Official source locations inspected:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_mmad_w4a4.hpp:440` through `:459`:
  official GMM2 copies the per-channel scale into the Fixpipe buffer and calls `copyL0CToGm(...)`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:196`
  and `:198`: `BlockEpilogue2` casts the official FP16 D2 high/low halves back to FP32.
- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py` now records `d2_half_variant_diagnostics` under
  `raw_c2_unfused_reference`.

High D2 half variant probe:

- Sentinel: `--swiglu-limit 451111`
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_high_half_rounding_variants_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_high_half_rounding_variants_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_high_half_rounding_variants_top1_expert0_max64.json`
- Probe exit: `1`, expected because strict comparator failed.
- Strict Gate B comparator remains failed:
  - `max_abs: 0.0009765625`
  - `mean_abs: 0.00004492711741477251`
  - failed elements over strict abs tolerance: `9010`
  - NaN/Inf counts: `0`
- Best variant by `(max_abs, mean_abs, failed_count)`:
  - `scale_low32_fp16_none`
  - key `[0.0009606480598449707, 0.000049416819820180535, 4087]`
- Best mean-error variant:
  - `scale_low32_fp16_toward_zero`
  - `max_abs: 0.0009765625`, `mean_abs: 0.000023340806365013123`, failed count `4840`
- High32 scale variants are rejected: best high32 rows have `max_abs: 1.2626953125`,
  `mean_abs: 0.1390603929758072`, failed count `130042`.

Low D2 half variant probe:

- Sentinel: `--swiglu-limit 453333`
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_low_half_rounding_variants_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_low_half_rounding_variants_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_low_half_rounding_variants_top1_expert0_max64.json`
- Probe exit: `1`, expected because strict comparator failed.
- Strict Gate B comparator remains failed:
  - `max_abs: 0.001953125`
  - `mean_abs: 0.00014362166984938085`
  - failed elements over strict abs tolerance: `38381`
  - NaN/Inf counts: `0`
- Best variant:
  - `scale_low32_fp16_toward_zero`
  - key `[0.001953125, 0.00007615549839101732, 20533]`
- High32 scale variants are rejected: best high32 rows have `max_abs: 3.37109375`,
  `mean_abs: 0.4442445635795593`, failed count `130747`.

Current conclusion:

- Low32 scale-word interpretation remains the only plausible scale-bit interpretation among the tested
  official packed scale words.
- Simple FP16 rounding-mode substitution improves some mean-error metrics but does not close max error or
  strict failed-element count for either half.
- Gate B remains failed. Gate C remains blocked until the official D2 half/Fixpipe contract is matched.
- The next useful diagnostic is lower than the current D2 half readback: expose or otherwise prove the exact
  official accumulator-to-Fixpipe dequant behavior or the concrete `CopyL0CToGm`/Fixpipe configuration used
  by the CATLASS tile copy, without changing the official GMM2 lifecycle.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check`: passed.
- Four-visible-NPU high-half and low-half rounding/scale variant probes completed and wrote summaries, both
  with expected nonzero exits because `passed: false`.

## Stage 2.2 D2 Half Boundary Diagnostic - 2026-06-26T21:10Z

This section is historical evidence. The `2026-06-26T21:35Z` section above supersedes it for current
status. Older sections are historical evidence unless explicitly
referenced here. It incorporates the binding requirements from
`svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Current Stage 2.2 top-1 runs still use the validated packed hidden INT4 and hidden-scale boundary. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate B D2 high and low half readbacks are finite and nonzero, but both fail the strict official-contract half-reference comparator. Gate C remains blocked by the still-failing Gate B numerical contract. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

This attempt is diagnostic-only. It does not use public `torch_npu.npu_grouped_matmul`, does not repack
weights, does not guess scale formulas, does not relax tolerances, and does not enable the production
SVDQ fused operator. The only device-code deviation is an isolated debug tap inside the official
`BlockEpilogue2` path to copy either the high or low FP16 D2 half after the official GM load and FP32 cast,
before `high * 16 + low`, aux add, hidden-scale multiply, BF16 cast, or final output routing.

Official source locations inspected for this attempt:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:794` keeps the
  full-lifecycle raw-debug sentinel range at `450000 < swigluLimit < 460000`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:800` maps in-range
  sentinels to debug modes: high D2 half, low D2 half, or combined `high * 16 + low`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:987` and
  `:1316` pass the debug mode into `BlockEpilogue2` for the GMM2-only and full-lifecycle dequant paths.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:157`
  computes the official high/low D2 GM offsets.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:196`
  casts the high FP16 D2 half to FP32; `:198` casts the low FP16 D2 half to FP32.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:200`
  and `:212` are the new debug-only high/low D2 copy taps.
- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py:695` implements the host-side official-contract
  D2 half comparator for this diagnostic; `:1177` maps the same sentinels on the probe side.

Official-vs-debug state table for the current D2-half attempt:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | Official mixed hidden producer, replaced at the debug boundary by Stage 2.1 packed INT4 bytes | Official GMM2 AIC reads A2 | Stage 2.1 host probe builds canonical hidden, INT8, packed INT4, then debug kernel copies it through `SeedGMM2OnlyPackedHidden` / full-lifecycle override | `gmA2I4_I8`, active rows by `problemShape.n()/2`; debug source tensor `hiddenXInt4Packed` | Expert-contiguous rows, 1024 packed bytes per row for hidden 2048 | Normal official lifecycle plus debug override | Before official GMM2 consumes A2 | GMM2 reads after override sync | Official GMM2 finalization unchanged | Gate A top-1 and top-k evidence remains exact for source and post-override packed bytes; no new hidden packing logic in this attempt |
| hidden scale `gmPerTokenScale2` | Official mixed hidden-scale producer, replaced at the debug boundary by Stage 2.1 per-token scale | `BlockEpilogue2` multiplies post-C2 FP32 by scale in normal post-dequant mode | Same debug boundary as packed hidden | `gmPerTokenScale2`, one FP32 scale per active row | Active rows | Normal official lifecycle plus debug override | Before normal AIV post-dequant | AIV reads after C2V wait | Official epilogue finalization unchanged | D2 half modes return before scale; scale is still recorded as Gate A metadata, not used in this half comparator |
| `tokenPerExpert` | Official routing/count path; debug top-1 supplies external counts | GMM2 scheduler and `BlockEpilogue2` row bounds | `SeedGMM2OnlyTokenState` / full-lifecycle token state | `tokenPerExpert` and peer token region through official layouts | Expert count vector, top-1 `[64,0,0,0,0,0,0,0]` | Official V2C/C2V lifecycle | Before GMM2 loop | GMM2 and CombineV2 consume counts | Official final drain unchanged | Gate A counts match active rows for these runs |
| `cumsumMM` | Official cumulative count helper | GMM2 scheduler | `GetCumsumForMMAIV` or single-EP copy path | Workspace `cumsumMM` | Expert prefix/cumulative counts | Official lifecycle | Before scheduling | GMM2 loop reads it | Official final drain unchanged | No new patch in this attempt |
| `preSumBeforeRank` | Official rank-prefix state | `BlockEpilogue2` / CombineV2 output offset logic | Official path initialization; debug single-rank zeroing in existing GMM2-only path | Workspace prefix state | Expert/rank prefix rows | Official lifecycle | Before CombineV2 | AIV consumes during epilogue | Official final drain unchanged | No new patch in this attempt |
| GMM2 AIC input tile state | Official `GMM2(params)` and `blockMmad` | Official MMAD/Fixpipe | GMM2 loop from official `DispatchFFNCombineW4A8` state | A2 packed hidden and W2 postloaded packed W4 | Official tiling | Official blockMmad flags | During GMM2 loop | AIC internal waits | `blockMmad.SynchronizeBlock` / `Finalize` | Unchanged; public grouped matmul is not used |
| GMM2 accumulator / D2 region | Official AIC MMAD/Fixpipe writes FP16 D2 high/low halves | `BlockEpilogue2` reads high and low halves | Workspace C2 binding from official kernel params | `gmCOffsetH = preSrcExpertSum*n2 + blockCoord.m()*n2 + blockCoord.n()` and `gmCOffsetL = gmCOffsetH + n2` | `LayoutC(actualBlockShape.m()/2, actualBlockShape.n(), params.n2 * 2)` for GM source | C2V handoff | After GMM2 tile completion | `CombineV2` waits C2V before `BlockEpilogue2` | Official finalization unchanged | New debug modes copy high or low FP32-cast half to `gmm2PostDequant` before combine math |
| C2V handoff state | Official GMM2 producer flags | Official AIV `CombineV2` / `BlockEpilogue2` | Official GMM2 loop | Official C2 workspace and sync flags | Tile order from official scheduler | C2V flag | After GMM2 tile output ready | `CombineV2` wait before epilogue tile | Official drain unchanged | D2 half taps occur after this wait, so finite/nonzero readback is Gate B boundary evidence |
| `BlockEpilogue2` input state | Official `CombineV2` passes `gmC2`, scale, aux, tile coord, group, and prefix state | `BlockEpilogue2` performs high/low decode, aux, scale, debug copy, and output | Epilogue params at kernel `:987` / `:1316` | Official `gmC2`, `gmPerTokenScale2`, `ptrMAux2`, debug output pointer | Active tile shape by `n0` and `n2` | UB events plus `EVENT_ID7` under `W4A8_DEBUG` | After C2V wait | UB MTE/V waits in `BlockEpilogue2` | `BlockEpilogue2::Finalize` | Only diagnostic branch changes return early for mode 2 or 3 |
| FP32 post-dequant debug tap | Official W4A8 debug output path | Host probe reads `gmm2PostDequant` | `W4A8_DEBUG` output pointer | `gmGMM2` offset `(preSrcExpertSum*n2 + blockCoord.m()*n2)/2 + blockCoord.n()` | Active rows by hidden size | `EVENT_ID7` | Copy around V/MTE3 event | Host reads after stream completion | Event wait in finalize | Mode 2 high-half and mode 3 low-half reuse this output; mode 1 remains combined raw C2; normal mode remains post-dequant |

Files changed in this attempt:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp`
  - Added `rawDebugMode` and debug-only D2 high/low half copy branches.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - Added `GMM2RawDebugMode(...)` and passed the mode to `BlockEpilogue2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/svdqw4_a8_gmm2_debug_readback.cpp`
  - Touched the debug entry source so the CANN source-copy target can be forced to track the D2-tap rebuild.
- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added the high/low D2 half reference and sentinel mapping.

Build and install evidence:

- Initial build command:
  `cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -j 8`
- Initial build log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_half_debug_build_kernel_forced_source_touch.log`
- Finding: initial rebuild did not refresh the generated source copy or `.o` files; object timestamps stayed at
  `2026-06-26T19:21Z` through `19:27Z`.
- Forced source-copy/generation refresh:
  removed the debug op source-copy stamp and per-shape generation stamps, then rebuilt the same target.
- Forced rebuild log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_half_debug_build_kernel_forced_stamp_refresh.log`
- Result: generated source copy timestamps refreshed to `2026-06-26T20:58Z`; all eight kernel object
  timestamps refreshed to `2026-06-26T21:00Z`.
- Install command copied `csrc/build/binary/ascend910b/bin/svdqw4_a8_gmm2_debug_readback/*` into the
  repo-local custom OPP tree.
- Source/installed SHA256 log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_half_debug_manual_kernel_refresh_forced_sha256.log`
- Result: source and installed hashes match.

Real-device commands preserved `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, used repo-local
`ASCEND_CUSTOM_OPP_PATH`, and used repo-local `op_api/lib` in `LD_LIBRARY_PATH`.

High D2 half probe:

- Sentinel: `--swiglu-limit 451111`
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_high_half_forced_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_high_half_forced_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_high_half_forced_top1_expert0_max64.json`
- Probe exit: `1`, expected because strict comparator failed.
- Gate B status: finite `true`, nonzero `true`, NaN count `0`, Inf count `0`.
- Comparator: `passed: false`, `max_abs: 0.0009765625`, `mean_abs: 0.00004492711741477251`,
  failed elements over strict abs tolerance `9010`, max relative error `0.001700680237263441`,
  mean relative error `0.0003222145023755729`.
- Absolute-error quantiles: q0.5 `0.0`, q0.9 `0.0001220703125`, q0.95 `0.000244140625`,
  q0.99 `0.000244140625`, q0.999 `0.00048828125`, q1 `0.0009765625`.

Low D2 half probe:

- Sentinel: `--swiglu-limit 453333`
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_low_half_forced_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_d2_low_half_forced_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_d2_low_half_forced_top1_expert0_max64.json`
- Probe exit: `1`, expected because strict comparator failed.
- Gate B status: finite `true`, nonzero `true`, NaN count `0`, Inf count `0`.
- Comparator: `passed: false`, `max_abs: 0.001953125`, `mean_abs: 0.00014362166984938085`,
  failed elements over strict abs tolerance `38381`, max relative error `0.0017152659129351377`,
  mean relative error `0.00032288453076034784`.
- Absolute-error quantiles: q0.5 `0.0`, q0.9 `0.00048828125`, q0.95 `0.00048828125`,
  q0.99 `0.0009765625`, q0.999 `0.001953125`, q1 `0.001953125`.

Current conclusion:

- Gate B is no longer an all-zero boundary: both official D2 halves are finite and nonzero after the official
  C2V wait and `BlockEpilogue2` GM load/cast boundary.
- Stage 2.2 still fails because neither D2 half passes the strict official-contract reference. Gate C must
  remain blocked until Gate B passes.
- The rejected hypothesis is "the residual is only caused by high*16+low recombination or hidden-scale
  post-dequant math"; the mismatch is already visible at each separate D2 half boundary.
- The next unresolved boundary is the exact official Fixpipe/FP16 rounding and packed-W4 D2 storage contract,
  while preserving the same official lifecycle and without using public grouped matmul or a substitute GEMM.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check`: passed.
- Forced CANN debug target rebuild completed and refreshed the generated source and object files.
- Source/installed custom OPP debug kernel hashes match.
- Four-visible-NPU high-half and low-half probes completed and wrote summaries, both with expected nonzero
  exits because `passed: false`.

## Stage 2.2 Residual Distribution Diagnostic - 2026-06-26T20:31Z

This section is historical evidence. The `2026-06-26T21:10Z` section above supersedes it for current
status. Older sections are historical evidence unless explicitly
referenced here.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Raw-C2 and post-dequant summaries now include residual distributions. The post-dequant residual is tightly bounded after hidden-scale application, but the strict Gate C max tolerance still fails; Gate B raw-C2 strict tolerance also still fails. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

This attempt is diagnostic-only. It does not modify device kernel code, synchronization, GMM2 lifecycle,
packed-weight access, scale formulas, public grouped matmul, tolerance values, or production SVDQ host tiling.

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added `_residual_distribution_diagnostics(...)`.
  - Raw-C2 and post-dequant reference blocks now report signed residual stats, absolute-error quantiles,
    threshold counts, top absolute-error locations, and row/column maxima.
  - Pass/fail logic and tolerances are unchanged.

Raw-C2 residual distribution probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and
  `--swiglu-limit 454545`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_residual_distribution_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_residual_distribution_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_residual_distribution_top1_expert0_max64.json`
- Top-level stage `passed: false`; this remains expected because the strict Gate B raw-C2 numerical gate
  still fails.

Raw-C2 residual evidence:

- error: `max_abs: 0.0166015625`, `mean_abs: 0.0007681758143007755`, failed elements over strict abs
  tolerance: `70547`
- signed residual mean: `0.00000616212491877377`
- signed counts: positive `41238`, negative `40921`, zero `48913`
- absolute-error quantiles:
  - q0.5: `0.000244140625`
  - q0.9: `0.00201416015625`
  - q0.95: `0.0037841796875`
  - q0.99: `0.00439453125`
  - q0.999: `0.0079345703125`
  - q1: `0.0166015625`
- threshold counts:
  - `abs_le_0.0002: 60525`, `abs_gt_0.0002: 70547`
  - `abs_le_0.001: 100098`, `abs_gt_0.001: 30974`
  - `abs_le_0.01: 131065`, `abs_gt_0.01: 7`
- top raw-C2 residual locations include `[44,1468]` with abs diff `0.0166015625`,
  `[54,1758]` with abs diff `0.01611328125`, and `[37,578]` with abs diff `0.0156402587890625`.

Post-dequant residual distribution probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, and `max_output_size=64`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_post_dequant_residual_distribution_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_post_dequant_residual_distribution_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_post_dequant_residual_distribution_top1_expert0_max64.json`
- Top-level stage `passed: false`; Gate C remains finite/nonzero and mean-error passes the old threshold,
  but max-error still exceeds the strict threshold.

Post-dequant residual evidence:

- error: `max_abs: 0.00037679076194763184`, `mean_abs: 0.000018463411834090948`
- signed residual mean: `0.0000001488513277081438`
- signed counts: positive `41238`, negative `40921`, zero `48913`
- absolute-error quantiles:
  - q0.5: `0.000005893409252166748`
  - q0.9: `0.00005221366882324219`
  - q0.95: `0.00008362531661987305`
  - q0.99: `0.00011137127876281738`
  - q0.999: `0.00019773695385083556`
  - q1: `0.00037679076194763184`
- threshold counts:
  - `abs_le_0.0002: 130953`, `abs_gt_0.0002: 119`
  - `abs_le_0.001: 131072`, `abs_gt_0.001: 0`
- top post-dequant residual locations include `[7,871]` with abs diff `0.00037679076194763184`,
  `[37,578]` with abs diff `0.0003540515899658203`, and `[49,1312]` with abs diff
  `0.00035199522972106934`.

Current conclusion:

- The post-dequant residual is much smaller than the raw-C2 residual after official AIV hidden-scale
  application, but strict Stage 2.2 numerical gates still fail.
- The signed residual distribution is balanced and shares the same positive/negative/zero counts between
  raw-C2 and post-dequant summaries, which is consistent with the same underlying C2 residual being scaled
  through the AIV path.
- This does not prove an accepted rounding model and does not justify tolerance relaxation. The next useful
  numerical step is a stricter official D2/Fixpipe boundary diagnostic, such as exposing high/low D2 halves
  or otherwise proving the exact device-side FP16 rounding contract without changing the official lifecycle.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check`: passed.
- Four-visible-NPU raw-C2 residual-distribution probe completed and wrote the new summary, with expected
  nonzero exit because `passed: false`.
- Four-visible-NPU post-dequant residual-distribution probe completed and wrote the new summary, with expected
  nonzero exit because `passed: false`.

## Stage 2.2 Multi-Expert Gate A Boundary Manifest - 2026-06-26T20:22Z

This section is historical evidence. The `2026-06-26T20:31Z` section above supersedes it for current
status. Older sections are historical evidence unless explicitly
referenced here.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Top-k 2 / experts 0 and 1 now records a multi-expert Gate A boundary manifest. It proves expert-contiguous GMM2 input ordering and post-override byte equality for the synthetic debug boundary, but it does not prove same source-token payload duplication across top-k slots. Gate B numerical parity still fails. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

This attempt is still host-side probe/report work only. It does not modify device kernel code, synchronization,
GMM2 lifecycle, packed-weight access, scale math, public grouped matmul, or production SVDQ host tiling.

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added `routing_identity.manifest_scope` so top-k greater than 1 summaries distinguish GMM2 boundary row
    identity from full upstream router token-duplication identity.

Multi-expert raw-C2 Gate A probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, `--top-k 2`, `--route-experts 0 1`, 32 source tokens, 64 active rows,
  `max_output_size=64`, and `--swiglu-limit 454545`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_gate_a_manifest_top2_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_gate_a_manifest_top2_expert0_1_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_gate_a_manifest_top2_expert0_1_max64.json`
- Top-level stage `passed: false`; this remains expected because the strict Gate B numerical reference still
  fails.

Multi-expert Gate A evidence:

- `active_expert_ids: [0,1]`
- `expert_token_nums: [[32,32,0,0,0,0,0,0]]`
- expert prefix sums first 8: `[0,32,64,64,64,64,64,64]`
- route slot mapping: slot 0 -> expert 0, slot 1 -> expert 1
- `expert_token_total_matches_active_rows: true`
- `reference_group_counts_match_expert_token_nums: true`
- source packed-hidden active SHA256:
  `a29851821d694cc55a182d5082fa535f5171acc5c7f3cda1fabfa7888ebd64de`
- post-override readback packed-hidden active SHA256:
  `a29851821d694cc55a182d5082fa535f5171acc5c7f3cda1fabfa7888ebd64de`
- padded rows: `padded_row_count: 0`, hidden INT4 padded nonzero count `0`, hidden-scale padded nonzero count `0`

Representative routed-row map:

- rows 0-31 map to expert 0, top-k slot 0, expert-local offsets 0-31.
- row 30 maps to source token 30, slot 0, expert 0, local offset 30.
- row 31 maps to source token 31, slot 0, expert 0, local offset 31.
- row 32 maps to source token 0, slot 1, expert 1, local offset 0.
- row 33 maps to source token 1, slot 1, expert 1, local offset 1.

Important limitation:

- `same_source_token_payload_across_topk_slots_proven: false`
- `token_major_order_matches_expert_contiguous_order: false`
- Interpretation: this synthetic Stage 2.2 probe now proves the external hidden boundary rows are consumed by
  official GMM2 in the declared expert-contiguous order and that post-override readback matches those bytes.
  It does not prove that the upstream router duplicated the same original token payload into every top-k slot.
  That remains a separate same-routing composition gate before production enablement.

Multi-expert raw-C2 numerical status:

- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: false`
- raw-C2 error: `max_abs: 16.85986328125`, `mean_abs: 1.1582927703857422`,
  failed elements over strict abs tolerance: `100825`

Current conclusion:

- Gate A boundary evidence now covers both top-1 single-expert and top-k 2 multi-expert expert-contiguous
  GMM2 input ordering for the isolated debug boundary.
- Stage 2.2 remains failed. The multi-expert run reinforces that row-order evidence is not enough; Gate B
  numerical parity still requires resolving the official D2/Fixpipe/reference residual without changing the
  official lifecycle.
- The same-source-token, same-routing full composition gate remains open and must be validated with a router
  path that carries duplicated token identity through top-k expansion.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- Four-visible-NPU top-k 2 raw-C2 probe completed and wrote the new manifest summary, with expected nonzero
  exit because `passed: false`.

## Stage 2.2 Gate A Routing Manifest Probe - 2026-06-26T20:15Z

This section is historical evidence. The `2026-06-26T20:22Z` section above supersedes it for current
status. Older sections are historical evidence unless explicitly
referenced here.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | The probe now emits a Gate A routed-row manifest with checksums, prefixes, expert-local offsets, and padded-row interpretation. Top-1 expert-0 row identity is proven for the current raw-C2 and post-dequant runs, but Gate B/Gate C numerical thresholds still fail and multi-expert row ordering remains to be proven. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

This attempt is a host-side probe/report improvement only. It does not modify the device kernel, does not
change synchronization, does not repack weights, does not use public grouped matmul, and does not enable
production SVDQ.

Files changed:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added SHA256/sample-byte manifest helpers for active hidden packed INT4 and hidden-scale boundaries.
  - Added `_routing_identity_manifest(...)` and wired it into loop-stats, raw-C2, and post-dequant summaries.
  - The manifest records active expert IDs, `expert_token_nums`, expert prefix sums, expert-local row starts
    and offsets, route slot to expert mapping, routed-row map samples, source/readback checksums, and
    padded-row zero interpretation.

Raw-C2 Gate A manifest probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and
  `--swiglu-limit 454545`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_gate_a_manifest_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_gate_a_manifest_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_gate_a_manifest_top1_expert0_max64.json`
- Top-level stage `passed: false`; this remains expected because the strict Gate B raw-C2 numerical gate
  still fails.

Raw-C2 Gate A evidence:

- `active_expert_ids: [0]`
- `expert_token_nums: [[64,0,0,0,0,0,0,0]]`
- expert prefix sums first 8: `[0,64,64,64,64,64,64,64]`
- `expert_token_total_matches_active_rows: true`
- `reference_group_counts_match_expert_token_nums: true`
- `token_major_order_matches_expert_contiguous_order: true` for this top-1 case
- first routed rows map row 0/1/2/3 to source token 0/1/2/3, top-k slot 0, expert 0, local offsets 0/1/2/3
- source packed-hidden active SHA256:
  `a29851821d694cc55a182d5082fa535f5171acc5c7f3cda1fabfa7888ebd64de`
- post-override readback packed-hidden active SHA256:
  `a29851821d694cc55a182d5082fa535f5171acc5c7f3cda1fabfa7888ebd64de`
- padded rows: `padded_row_count: 0`, hidden INT4 padded nonzero count `0`, hidden-scale padded nonzero count `0`

Raw-C2 numerical status is unchanged:

- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_aic_reference_passed: false`
- raw-C2 error: `max_abs: 0.0166015625`, `mean_abs: 0.0007681758143007755`,
  failed elements over strict abs tolerance: `70547`

Post-dequant Gate A manifest probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, and `max_output_size=64`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_post_dequant_gate_a_manifest_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_post_dequant_gate_a_manifest_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_post_dequant_gate_a_manifest_top1_expert0_max64.json`
- Top-level stage `passed: false`; Gate C remains finite/nonzero but fails the strict max tolerance.

Post-dequant Gate A evidence matches raw-C2:

- `active_expert_ids: [0]`
- `expert_token_nums: [[64,0,0,0,0,0,0,0]]`
- `expert_token_total_matches_active_rows: true`
- `reference_group_counts_match_expert_token_nums: true`
- source and post-override readback packed-hidden active SHA256 both equal
  `a29851821d694cc55a182d5082fa535f5171acc5c7f3cda1fabfa7888ebd64de`
- padded-row nonzero counts remain zero

Post-dequant numerical status is unchanged:

- canonical hidden finite/nonzero: `true` / `true`
- post-dequant reference passed: `false`
- post-dequant error: `max_abs: 0.00037679076194763184`, `mean_abs: 0.000018463411834090948`

Current conclusion:

- Gate A is materially stronger for the current top-1 expert-0 probe: row identity, prefixes,
  expert-local offsets, boundary checksums, and padding are now recorded in the summary.
- Stage 2.2 is not complete. Gate B and Gate C strict numerical gates still fail.
- The next unresolved Gate A gap is multi-expert/top-k routed-row ordering. The next numerical gap is still
  the official FP16 D2/Fixpipe residual between raw-C2 readback and the host reference.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- Four-visible-NPU raw-C2 probe completed and wrote the new manifest summary, with expected nonzero exit
  because `passed: false`.
- Four-visible-NPU post-dequant probe completed and wrote the new manifest summary, with expected nonzero
  exit because `passed: false`.

## GMM2 Official-Path Appendix Adoption - 2026-06-26T20:01Z

This section is historical evidence. The `2026-06-26T20:15Z` section above supersedes it for current
status. It incorporates
`/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`
as binding for subsequent work.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate B and Gate C are finite/nonzero after the official `zN`/FP16 D2 reference correction, but strict numerical gates still fail. The new GMM2 official-path appendix now blocks speculative lifecycle patches until the official-vs-debug state table and missing routing identity evidence are complete. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Status reconciliation:

- The appendix stop conditions and acceptance criteria are binding for future implementation.
- The appendix statement that the remaining failure is an all-zero post-dequant tensor is stale relative to
  commit `df3aab3d2497b789348fe987e1fbcbf552b865cd`: the latest corrected raw-C2 and post-dequant probes
  are finite and nonzero, but their strict max-error gates still fail.
- No kernel, synchronization, state initialization, layout, or public grouped-matmul change was made for this
  adoption section.
- Source inspection, manifests, scalar substitutes, public grouped-matmul experiments, compilation, kernel
  launch, and `official_gmm2_entry_reached=true` do not count as Stage 2.2 gate progress.

Binding next-step constraints:

- Do not modify V2C/C2V flag order, producer/consumer ownership, `tokenPerExpert`, `cumsumMM`,
  `preSumBeforeRank`, AIC/AIV roles, D2 source addresses, workspace offsets, or barriers unless the exact
  successful official W4A8 counterpart is cited and the deviation is recorded first.
- Preferred correction path is a known-successful official W4A8 lifecycle with only the GMM2 hidden packed
  input and hidden-scale boundary overridden by validated Stage 2.1 tensors.
- Gate A must be completed with routed-row identity, prefix sums, expert-local row starts/offsets, active-row
  and padded-row interpretation, metadata, checksums, and representative bytes.
- Gate B must expose and compare the official AIC GMM2 raw/D2 boundary.
- Gate C must validate `BlockEpilogue2` / `CombineV2` post-dequant output under the predeclared strict gate.

Official source locations rechecked for this adoption:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:284-314`:
  `cumsumMM`, `gmA2I4`, `gmA2I4_I8`, `gmC2`, `gmPerTokenScale2`, `tokenPerExpert`, and
  `preSumBeforeRank` bind to the official workspace/shared-memory regions.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:413-430`:
  `FetchAndPreprocessInt8ToInt4` is the official hidden INT8 to packed INT4 producer helper.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:684-775`:
  `GMM2` waits on `SYNCFLAGV2C`, derives per-expert `currentM` from `cumsumMM`, doubles M for INT4,
  uses official `layoutA2`, `layoutB2`, `layoutScale2`, and writes FP16 `gmC2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:930-991`:
  the isolated debug path seeds token state, packed hidden, and hidden scale, signals readiness, and invokes
  official `CombineV2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1318-1388`:
  the full official lifecycle waits C2V, runs `BlockEpilogue1`, optionally overrides the hidden GMM2 input
  boundary, signals V2C, debug-copies hidden readbacks, then runs `CombineV2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1422-1499`:
  `CombineV2` waits GMM2 completion flags, invokes `BlockEpilogue2`, and finalizes the epilogue.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:215-228`:
  `InitGMM2OnlyFromPacked` wires external hidden, scale, and expert-token inputs into the debug entry.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:249-314`:
  official W4A8 GMM2 uses `LayoutA=RowMajor`, W2 `LayoutB=layout::zN` when `Nz_` is true,
  `CType=float16_t`, and row-major `layoutD2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:154-239`:
  `BlockEpilogue2` reads adjacent high/low FP16 C2 rows, casts to FP32, computes `high * 16 + low`,
  optionally raw-debug copies the pre-hidden-scale value, then applies hidden per-token scale.

Expanded official-vs-debug state table required before the next behavioral patch:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | `BlockEpilogue1` through official hidden quant/pack; full-lifecycle debug may overwrite this boundary only after the producer barrier | GMM2 AIC as `gmA2I4`; debug readback also copies `gmA2I4_I8` | Workspace binding at kernel init; official producer during `DispatchAndCombine` C2V loop | `workspaceInfo.ptrA2Int4`; debug override copies external packed hidden to the same region at `gmOffsetD` | Active routed rows, physical bytes `curRowNum * problemShape.n()/2`; GMM2 sees doubled logical M | V2C | After `BlockEpilogue1` and optional hidden override | `GMM2` waits `SYNCFLAGV2C` before consuming group tiles | Official `blockEpilogue1.Finalize`, then `CombineV2`/reset | Exact post-override readback in latest probes; row identity evidence still incomplete. |
| hidden scale `gmPerTokenScale2` | `BlockEpilogue1`; full-lifecycle debug may overwrite with Stage 2.1 scale at same boundary | `BlockEpilogue2` post-dequant | Workspace binding at kernel init; produced per active routed row | `workspaceInfo.ptrPerTokenScale2`; debug override copies external scale at `rowStartThisCore` | One FP32 scale per routed row | V2C | Same as packed hidden | `BlockEpilogue2` reads during `CombineV2` after C2V wait | `blockEpilogue.Finalize` | Exact post-override max abs 0; prefix/row-offset proof still needs richer evidence. |
| `tokenPerExpert` | Official routing into peer shared memory | `GetCumsumForMMAIV`, epilogue routing metadata, final reset | Kernel init binds shared-memory peer token region | `shmem() + peermemInfo.offsetPeerTokenPerExpert`; layout uses aligned expert count | One count slot per EP/expert, aligned to 128 | Official shared state, then V2C/C2V users | Routing completes before GMM stages | GMM/epilogue consume after handoff waits | `ResetTokenPerExpert` after `CombineV2` | Top-1 expert-0 probe records `[64,0,0,0,0,0,0,0]`; multi-route row identity not yet proven. |
| `cumsumMM` | Official routing cumsum or debug `SeedGMM2OnlyTokenState` for EP=1 | GMM2 group loop and `CombineV2` group loop | Kernel init binds `workspaceInfo.ptrcumsumMM` | Workspace `ptrcumsumMM`; group index `(EP - 1) * expertPerRank + groupIdx` | One cumulative/current count per local expert for the last rank view | V2C/C2V-dependent consumers | Must be valid before GMM2 tiles launch | GMM2 waits V2C at group sync points | N/A, workspace reused after op | Loop-stats debug shows valid arrays for top-1; padded-row interpretation still incomplete. |
| `preSumBeforeRank` | Official prefix/rank setup | `BlockEpilogue2` for output placement | Kernel init binds `workspaceInfo.ptrSumBeforeRank` | Workspace `ptrSumBeforeRank` | Per expert/rank prefix | C2V-side consumer | Must be initialized before `CombineV2` | Epilogue consumes after GMM completion wait | N/A | Debug zeroes it for EP=1; report still needs explicit proof for EP/TP variants before production. |
| GMM2 AIC input tile state | `GMM2` constructs layouts from official params and workspace | `BlockMmad` | Inside `GMM2` per expert group | `gmA2I4[gmGroupOffsetA + layoutA.GetOffset(offsetA)]`, W2 from `GetTensorAddr(...ptrB2)`, scale from `ptrScale2` | `L1TileShape<128,256,1024>`; INT4 doubles M; W2 official `zN` | V2C before input use | AIV hidden producer signals V2C | AIC waits V2C before group tiles | `blockMmad.Finalize` later signals C2V | Latest corrected host reference matches official layout closely but strict raw-C2 gate still fails. |
| GMM2 accumulator / D2 region | Official W4A8 AIC `BlockMmad` with FP16 `CType` | `BlockEpilogue2` | `GMM2` per tile | `workspaceInfo.ptrC2`; `gmC2[gmGroupOffsetC + layoutC.GetOffset(offsetC)]` | Row-major C tile; high/low rows adjacent for epilogue | C2V | `BlockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V)` after tile production | `CombineV2` waits cross-core flag before epilogue | `blockEpilogue.Finalize` | Raw-C2 readback finite/nonzero; max error `0.0166015625`, mean `0.0007681758`, strict gate failed. |
| C2V handoff state | GMM2 AIC finalize | AIV `CombineV2` | During GMM2 tile loop/finalize | Cross-core flag IDs derived from group sync index | Per expert group / cross-core flag batch | C2V | After relevant AIC tiles complete | `CombineV2` waits flags before `BlockEpilogue2` | Wait loop drains remaining groups | Latest raw mode reports handoff verified; no bypass flags should be added. |
| `BlockEpilogue2` input state | GMM2 FP16 C2 plus hidden scale and MAux2 | FP32 post-dequant / final D2 output | `CombineV2` constructs `BlockEpilogue2` with official params | Reads `gmC2`, `gmPerTokenScale2`, `ptrMAux2`, writes D/output or debug `gmCGMM2` | M split by AIV subcore; high/low C2 rows combine as `high*16+low` before hidden scale | C2V and UB events | C2V must be complete before each tile | Waits C2V in `CombineV2`; UB events inside epilogue | `blockEpilogue.Finalize` | Post-dequant finite/nonzero; max error `0.00037679`, mean `0.00001846`; strict max gate failed. |
| FP32 post-dequant debug tap | `BlockEpilogue2` W4A8 debug path | Host probe summary | Inside `BlockEpilogue2` raw or normal debug output path | Debug output tensor `gmCGMM2` / op output | Active rows x hidden size | UB MTE/V events plus C2V precondition | After high/low combine and optionally after hidden-scale multiply | Host reads after op completion | Normal op completion | Useful diagnostic only; not a production substitute and not a pass condition without strict reference match. |

Required status fields for all future Stage 2.2 summaries:

- `official_gmm2_entry_reached`
- `official_gmm2_loop_count`
- `official_gmm2_active_tile_count`
- `official_gmm2_aic_raw_output_finite`
- `official_gmm2_aic_raw_output_nonzero`
- `official_gmm2_aic_reference_passed`
- `official_gmm2_c2v_handoff_verified`
- `official_gmm2_post_dequant_finite`
- `official_gmm2_post_dequant_nonzero`
- `official_gmm2_post_dequant_reference_passed`
- `official_gmm2_numerical_gate_passed`

Immediate unresolved boundary:

- Complete Gate A row-identity evidence beyond the current top-1 shape/count equality.
- Decide whether the remaining raw-C2 residual is explained by official Fixpipe/FP16 rounding, or expose a
  stricter device-side D2/pre-D2 boundary without changing the official lifecycle.
- Keep Stage 2.2 `FAIL / IN PROGRESS` and production fail-closed until `official_gmm2_numerical_gate_passed`
  is true on real Ascend 910B4 with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.

## Official zN / FP16 D2 Reference Correction - 2026-06-26T19:51Z

This section is historical evidence. The `2026-06-26T20:01Z` section above supersedes it for current
work constraints. Older sections are historical evidence unless explicitly
referenced here.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Correcting the host reference to official W2 `layout::zN` and FP16 D2 storage removes the previous large raw-C2/post-dequant mismatch, but the strict Gate B/Gate C thresholds still do not pass. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical gates. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints remain unchanged:

- Use only `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Do not use or debug public `torch_npu.npu_grouped_matmul`.
- Official `dispatch_ffn_combine_w4_a8` remains the only W4A8 source of truth.
- Do not advance to SVDQ down, final combine, or production enablement while Stage 2.2 fails.

This attempt is a host-reference correction and diagnostic extension only. It does not modify the device
kernel, does not repack checkpoint weights, does not change synchronization, does not use public grouped
matmul, and does not enable production SVDQ.

Files changed for this attempt:

- `tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py`
  - Added an official W4A8 C2 helper that models the actual D2 boundary:
    `fp16(high_acc * weight_scale) * 16 + fp16(low_acc * weight_scale)`.
  - Added a shared postloaded W4 `layout::zN::MakeLayout<int4b_t>` host unpack helper.
  - Updated GMM1/GMM2 unfused and raw-C2 references to use official postloaded W4 `zN` layout and FP16
    D2 storage before AIV post-processing.
- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added a diagnostic-only `catlass_weight_zN_official_b_layout` W2 variant.
  - Updated local raw-C2 layout variants to use the same FP16 D2 reference boundary.

Official source locations used for this correction:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:250-257`:
  `weightNz=true` selects `LayoutB=layout::zN`; `layoutB2` is created through
  `LayoutBInitializer<layout::zN, int4b_t>::create(k2, n2)`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/select_helper.hpp:17-23`:
  `LayoutBInitializer<layout::zN, int4b_t>` calls `layout::zN::MakeLayout<int4b_t>(k, n)`.
- `csrc/third_party/catlass/include/catlass/layout/matrix.hpp:519-536`:
  official `zN::MakeLayout` for int4 has `C0_NUM_PER_FRACTAL=16`, `ELE_NUM_PER_C0=64`, and the offset
  formula used by the corrected host reference.
- `csrc/third_party/catlass/include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp:430-456`:
  B operand copy keeps `layout::zN` in L1 and uses `LoadDataWithTranspose` for L0B.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:278`:
  W4A8 GMM `CType` is `float16_t`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:169-203`:
  `BlockEpilogue2` reads high/low FP16 rows from `gmC2`, casts them to FP32, computes `high * 16 + low`,
  and the raw-debug tap copies that FP32 value.
- `csrc/third_party/catlass/include/catlass/gemm/helper.hpp:138-139`:
  int4 x int4 accumulator type is `int32_t`.
- `csrc/third_party/catlass/include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp:220-248`:
  per-channel Fixpipe writes the scaled accumulator to GM as the configured `ElementDst`.

Corrected raw-C2 probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and
  `--swiglu-limit 454545`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_primary_zN_fp16_boundary_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_primary_zN_fp16_boundary_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_primary_zN_fp16_boundary_top1_expert0_max64.json`
- Top-level stage `passed: false`; Gate B still does not satisfy the old strict raw-C2 thresholds.

Raw-C2 corrected primary evidence:

- `official_gmm2_entry_reached: true`
- `official_gmm2_aic_raw_output_finite: true`
- `official_gmm2_aic_raw_output_nonzero: true`
- `official_gmm2_c2v_handoff_verified: true`
- hidden packed post-override exact-match: `true`, mismatch count `0`
- hidden scale post-override exact mismatch count `0`, max abs `0.0`
- corrected primary contract:
  `postloaded W4 weight interpreted with Catlass layout::zN::MakeLayout<int4b_t>`;
  `fp16(high_acc * postloaded_weight_scale) * 16 + fp16(low_acc * postloaded_weight_scale)`
- corrected primary raw-C2 error:
  - `max_abs: 0.0166015625`
  - `mean_abs: 0.0007681758143007755`
  - `failed_element_count_abs_gt_tolerance: 70547` / `131072`
  - NaN/Inf counts: zero for actual, expected, and diff
- previous row-major primary error, before the correction, was about `max_abs: 23.6257`,
  `mean_abs: 3.0853`; the large mismatch is now explained by the wrong host W2 layout and missing FP16 D2
  boundary in the reference.

W2 layout variants after the correction:

| Variant | Mean abs | Max abs | Failed elements over old abs tolerance |
|---|---:|---:|---:|
| current row-major host reference | `3.0852599143981934` | `23.625732421875` | `131060` |
| CATLASS `nZ` historical diagnostic | `3.1503801345825195` | `28.720703125` | `131057` |
| official CATLASS `zN` B layout | `0.0007681758143007755` | `0.0166015625` | `70547` |

Corrected post-dequant probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, and `max_output_size=64`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_post_dequant_primary_zN_fp16_reference_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_post_dequant_primary_zN_fp16_reference_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_post_dequant_primary_zN_fp16_reference_top1_expert0_max64.json`
- Top-level stage `passed: false`; Gate C is finite/nonzero and close, but the max error still exceeds the
  pre-existing strict tolerance.

Post-dequant corrected evidence:

- output finite: `true`
- output nonzero: `true`
- output active stats: `max_abs: 0.46528252959251404`, `mean_abs: 0.05298978090286255`
- corrected post-dequant reference error:
  - `max_abs: 0.00037679076194763184`
  - `mean_abs: 0.000018463411834090948`
  - configured max tolerance: `0.0002`
  - configured mean tolerance: `0.00002`
- Interpretation: the earlier all-zero or large-mismatch characterization is stale for the corrected
  official reference. The official path produces finite nonzero post-dequant output with row identity intact,
  but Stage 2.2 remains failed because the declared numerical gates still do not both pass.

Current conclusion:

- The main Stage 2.2 evidence gap moved from "why is raw C2 broadly wrong?" to "what exact numeric tolerance
  or device rounding model should be used for the official FP16 D2 boundary?"
- The corrected reference proves the official modified-hidden GMM2 path consumes the same routed rows and
  the official `zN` W2 layout, and produces finite nonzero raw and post-dequant outputs.
- Do not mark Stage 2.2 complete yet. The next work should either prove the residual is exactly explained by
  official Fixpipe/FP16 rounding semantics with a documented non-relaxed gate, or add a device-side readback
  that exposes the pre-FP16 accumulator / per-half D2 values to remove the remaining ambiguity.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check`: passed.
- Four-visible-NPU raw-C2 probe: completed on `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, wrote summary, and
  correctly reported Stage 2.2 `passed: false`.
- Four-visible-NPU post-dequant probe: completed on `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, wrote summary, and
  correctly reported Stage 2.2 `passed: false`.

## Full-Lifecycle Loop-State Diagnostic Hook - 2026-06-26T19:31Z

This section is historical evidence. The `2026-06-26T19:51Z` section above supersedes it for current status.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A is exact and the official GMM2 full-lifecycle loop-state probe now shows real token/cumsum state and active tiles, but strict GMM2 numerical parity remains failed. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 modified-hidden official W4A8 GMM2 numerical parity. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints remain unchanged:

- Use only `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Do not use or debug public `torch_npu.npu_grouped_matmul`.
- Official `dispatch_ffn_combine_w4_a8` remains the only W4A8 source of truth.
- Do not advance to SVDQ down, final combine, or production enablement while Stage 2.2 fails.

Appendix `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding.
This change is diagnostic-only. It does not modify the ordinary official W4A8 path, does not use public
grouped matmul, does not introduce a substitute GMM2 implementation, and does not enable production SVDQ
host tiling.

Problem found before this hook:

- The previous full-lifecycle loop-state probe ran with `--swiglu-limit 434343` but reported
  `official_gmm2_entry_reached: false`, `official_gmm2_loop_count: 0`, `active_tile_count: 0`,
  `loop_stats.valid_magic: false`, and zeroed token/cumsum state.
- Hidden packed override and hidden scale override were already exact, so the missing loop-state magic was
  an instrumentation/synchronization visibility gap rather than proof that Stage 2.2 numerical parity had
  advanced.

Files changed for this attempt:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - In `GMM2`, the existing loop-stats sentinel now waits on the official `SYNCFLAGV2C`, writes
    `WriteGMM2OnlyLoopStats(params)`, and returns before issuing GMM2 tiles.
  - In `DispatchAndCombine`, the same loop-stats sentinel returns before `CombineV2` so AIV does not wait
    for C2V flags that this diagnostic mode intentionally does not produce.
  - The hook is gated by the existing loop-stats debug predicate only; normal W4A8, raw-C2 debug mode, and
    production SVDQ are unchanged.

Official source locations for this diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/svdqw4_a8_gmm2_debug_readback.cpp`:
  debug wrapper calls `InitGMM2OnlyFromPacked(...)` and then `Process()`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h`:
  `InitGMM2OnlyFromPacked` currently runs the full official lifecycle with external hidden override by
  setting `gmm2OnlyFromPacked_ = false`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:704-708`:
  AIC loop-stats hook waits for the official V2C handoff and writes loop stats.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1382-1384`:
  AIV loop-stats hook returns before `CombineV2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`:
  ordinary GMM2 still consumes `gmA2I4` and `gmB2` through the official layouts and writes `gmC2`;
  ordinary `BlockEpilogue2` still owns dequantization and output scatter outside this diagnostic sentinel.

Official-vs-debug state table:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | Official `BlockEpilogue1` after GMM1 SwiGLU quant; debug override may replace after the official producer barrier | Official GMM2 AIC `gmA2I4` | `DispatchAndCombine` full lifecycle | Official D1/A2 GM region; debug override copies external packed hidden into this same region | D1 physical bytes equal A2 doubled-M INT4 bytes for active rows | V2C notify before GMM2 | After all AIV cores finish, then core 0 external copy, then post-copy barrier | GMM2 waits on official `SYNCFLAGV2C` | Official lifecycle outside loop-stats sentinel | Exact in latest run: mismatch count 0 |
| hidden scale `gmPerTokenScale2` | Official `BlockEpilogue1`; debug override may replace after the official producer barrier | Official `BlockEpilogue2` / post-dequant | `DispatchAndCombine` full lifecycle | Official per-token scale GM region; debug override copies external scale into same region | One FP32 scale per active row | Same V2C lifecycle | Same as packed hidden | Consumed after GMM2 raw C2 | Official lifecycle outside loop-stats sentinel | Exact in latest run: exact mismatch count 0, max abs 0 |
| routing state `tokenPerExpert` / `cumsumMM` | Official AIV routing and `GetCumsumForMMAIV` | Official GMM1/GMM2 AIC group loops | Before GMM1 hidden producer and before V2C signal | Official workspace routing arrays | One count per expert/rank | V2C dependency for GMM2 loop-stats hook | AIV prepares state before V2C | New diagnostic AIC hook waits V2C | No C2V drain in loop-stats mode | Valid magic now true; external/token/cumsum arrays match `[64,0,0,0,0,0,0,0]` |
| W2 packed weight `gmB2` | ModelSlim post-load `maybe_trans_nz` W2 | Official GMM2 AIC `gmB2` | `process_weights_after_loading_modelslim` | `GetTensorAddr<int4b_t>(arrayGroupIdx, params.ptrB2)` | `LayoutB=layout::zN`, `int4b_t`, `weightNz=true` | none | Static input | GMM2 tile loop | Official lifecycle | Previous W2 NZ diagnostic rejected a simple host W2 layout mismatch as the primary root cause |
| GMM2 accumulator / D2 region | Official GMM2 AIC | Official C2V / `BlockEpilogue2` | GMM2 loop | `gmC2[gmGroupOffsetC + layoutC.GetOffset(offsetC)]` | Row-major C tile layout | C2V handoff | Ordinary GMM2 notifies after tile production | AIV waits before epilogue | Official drain | Loop-stats sentinel intentionally does not produce C2V; raw-C2 numerical gate remains unresolved |
| FP32 post-dequant debug tap | Official `BlockEpilogue2` raw-debug path or W4A8 debug tap | Host readback | `W4A8_DEBUG` path | Existing debug output GM | Active rows x hidden size | C2V handoff | After ordinary GMM2 raw C2 | AIV reads C2 | Official lifecycle | Gate C remains blocked while Gate B raw-C2 parity fails |

Build/install evidence:

- Focused debug kernel rebuild:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_full_lifecycle_loop_stats_build_kernel.log`
  completed with `Built target svdqw4_a8_gmm2_debug_readback_ascend910b`.
- Repo-local installer attempt:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_full_lifecycle_loop_stats_install_repo_opp.log`
  reported an uninstall-script copy error and did not refresh the kernel artifacts.
- Manual repo-local debug-kernel artifact refresh:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_full_lifecycle_loop_stats_manual_kernel_refresh_sha256.log`
  shows all eight installed `SVDQW4A8GMM2DebugReadback_*.o` hashes match the freshly generated build outputs.

Loop-state diagnostic probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and
  `--swiglu-limit 434343`.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_full_lifecycle_loop_stats_after_hook_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_full_lifecycle_after_hook_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_loop_stats_full_lifecycle_after_hook_top1_expert0_max64.json`
- Top-level stage `passed: false`; this is expected because the loop-state diagnostic does not satisfy the
  numerical Gate B/Gate C requirements.

Latest extracted evidence:

- `official_gmm2_entry_reached: true`
- `official_gmm2_loop_count: 8`
- `official_gmm2_active_tile_count: 8`
- `official_gmm2_active_tile_count_nonzero: true`
- hidden packed post-override readback exact-match: `true`, mismatch count `0`
- hidden scale post-override readback exact mismatch count `0`, max abs `0.0`
- routing identity: expert-token counts `[[64,0,0,0,0,0,0,0]]`, reference group counts
  `[64,0,0,0,0,0,0,0]`
- loop stats: `valid_magic: true`, `magic: 434343.0`, `expert_per_rank: 8`, `max_output_size: 64`,
  `total_active_rows: 64`, `total_doubled_rows: 128`, `total_core_loops: 8`, `groups_with_work: 1`,
  `final_pre_current_m_sum: 64`, `n2: 2048`, `k2: 512`, `core_num: 20`
- group 0: `raw_current_m: 64`, `clipped_current_m: 64`, `doubled_current_m: 128`, `core_loops: 8`,
  `pre_current_m_sum: 0`
- groups 1-7: zero rows and zero core loops, with `pre_current_m_sum: 64`
- state probe: `valid_magic: true`, `ep: 1`, `rank: 0`, `layout_base_equals_cumsum_base: true`
- state probe arrays:
  - `external_expert_token_nums_first32`: `[64,0,0,0,0,0,0,0]`
  - `token_per_expert_cumsum_base_first32`: `[64,0,0,0,0,0,0,0]`
  - `token_per_expert_layout_base_first32`: `[64,0,0,0,0,0,0,0]`
  - `cumsum_mm_last_rank_first32`: `[64,0,0,0,0,0,0,0]`

Current conclusion:

- The diagnostic hook fixed the missing official-loop-state readback for the full-lifecycle modified-hidden
  debug op.
- This is Gate A/state evidence only. It does not close Stage 2.2 because the ordinary raw-C2 numerical
  comparator remains failed from the earlier raw-C2 probes.
- The next Stage 2.2 work should use this now-valid token/cumsum/tile-state evidence to continue isolating
  the official GMM2 accumulator/D2 mapping, C2V handoff, and `BlockEpilogue2` source mapping. It should not
  restart public grouped-matmul experiments or add SVDQ BF16 projections.

Validation before this report update:

- Focused debug kernel build: passed.
- Repo-local debug kernel artifact hashes: refreshed and matched build outputs.
- Four-visible-NPU loop-state probe: completed on `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, wrote summary, and
  correctly reported Stage 2.2 `passed: false` because numerical parity is still unresolved.

## W2 NZ Layout Comparator Diagnostic - 2026-06-26T19:00Z

This section is historical evidence. The `2026-06-26T19:31Z` section above supersedes it for current status.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Gate A remains exact and Gate B raw C2 remains finite/nonzero, but strict raw-C2 parity still fails. The new W2 `weightNz=true` layout diagnostic rejects a simple host W2 layout comparator mismatch as the primary explanation. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 raw-C2 numerical parity. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints remain unchanged:

- Use only `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Do not use or debug public `torch_npu.npu_grouped_matmul`.
- Official `dispatch_ffn_combine_w4_a8` remains the only W4A8 source of truth.
- Do not advance to SVDQ down, final combine, or production enablement while Stage 2.2 fails.

Appendix `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding.
This attempt is a Python-side diagnostic extension only. It does not change device behavior, GMM2
synchronization, public grouped matmul usage, production host tiling, or the official hidden/scale
override boundary.

Files changed for this attempt:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added `raw_c2_weight_layout_variant_diagnostics`.
  - Added a diagnostic CATLASS `nZ` / `zN` INT4 W2 unpack based on the official `weightNz=true` source path.
  - Kept the existing row-major Gate B comparator and reports both variants side by side.
  - Marks the new host-side computation as diagnostic-only, not a substitute GMM2 implementation or a proposed repack.

Official source locations inspected for this diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_gmm2_debug_readback.cpp:44-48`:
  debug GMM2 op API uses `transB=false` and `weightNz=true`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:309-314`:
  `LayoutB` is `layout::zN` when `weightNz=true`; GMM2 creates `layoutB2` for `int4b_t`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:690-724`:
  GMM2 sets `n2=k`, `k2=n/2`, and doubles `currentM` for INT4.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:752-767`:
  GMM2 consumes `gmA2I4` and `gmB2` through the official layouts and writes `gmC2`.
- `csrc/third_party/catlass/include/catlass/layout/matrix.hpp:345-365`:
  CATLASS `nZ::MakeLayout` and `GetOffset` formula for INT4 W2 access.
- `vllm_ascend/quantization/methods/w4a8.py:739-740`:
  ModelSlim post-load applies `maybe_trans_nz` to `w13_weight` and `w2_weight`.
- `vllm_ascend/quantization/methods/svdq_post_load.py:154-160`:
  helper records NPU format metadata where available.

Official-vs-debug state-table update:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | Official `BlockEpilogue1` after GMM1 SwiGLU quant | Official GMM2 AIC `gmA2I4` | `DispatchAndCombine` full lifecycle | Official D1/A2 GM region; debug override copies external packed hidden into this same region | D1 physical bytes equal A2 doubled-M INT4 bytes for active rows | V2C notify before GMM2 | After all AIV cores finish, then core 0 external copy, then post-copy barrier | GMM2 waits on official `SYNCFLAGV2C` | Official lifecycle | Exact in latest run: mismatch count 0 |
| hidden scale `gmPerTokenScale2` | Official `BlockEpilogue1` | Official `BlockEpilogue2` / post-dequant | `DispatchAndCombine` full lifecycle | Official per-token scale GM region; debug override copies external scale into same region | One FP32 scale per active row | Same V2C lifecycle | Same as packed hidden | Consumed after GMM2 raw C2 | Official lifecycle | Exact in latest run: exact mismatch count 0, max abs 0 |
| W2 packed weight `gmB2` | ModelSlim post-load `maybe_trans_nz` W2 | Official GMM2 AIC `gmB2` | `process_weights_after_loading_modelslim` | `GetTensorAddr<int4b_t>(arrayGroupIdx, params.ptrB2)` | `LayoutB=layout::zN`, `int4b_t`, `weightNz=true` | none | Static input | GMM2 tile loop | Official lifecycle | New diagnostic compares current row-major host comparator vs CATLASS `nZ` host interpretation |
| GMM2 accumulator / D2 region | Official GMM2 AIC | Official C2V / `BlockEpilogue2` | GMM2 loop | `gmC2[gmGroupOffsetC + layoutC.GetOffset(offsetC)]` | Row-major C tile layout | C2V handoff | GMM2 notifies after tile production | AIV waits before epilogue | Official drain | Raw C2 finite/nonzero but numerical parity fails |
| FP32 post-dequant debug tap | Official `BlockEpilogue2` raw-debug path or W4A8 debug tap | Host readback | `W4A8_DEBUG` path | Existing debug output GM | active rows x hidden size | C2V handoff | After GMM2 raw C2 | AIV reads C2 | Official lifecycle | Gate C remains blocked while Gate B raw-C2 parity fails |

Raw-C2 W2 NZ layout diagnostic probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and
  `--swiglu-limit 454545`.
- Preflight `npu-smi info` showed physical NPUs 0-3 as 910B4 with no running NPU processes.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_w2_nz_layout_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_w2_nz_layout_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_w2_nz_layout_top1_expert0_max64.json`
- Top-level stage `passed: false`; Stage 2.2 remains failed.
- Runtime environment: `npu_device_count: 4`, selected device `0`, runtime SOC `Ascend910B4`.
- `production_svdq_host_tiling_fail_closed: true`.

Gate A evidence:

- Source packed hidden exact-match: `hidden_packed_exact: true`, mismatch count zero.
- Post-override packed hidden readback: `exact_match: true`, mismatch count `0`.
- Post-override hidden scale readback: `exact_mismatch_count: 0`, `max_abs: 0.0`.
- Canonical hidden and hidden scale are finite and nonzero.
- W2 metadata: shape `[8, 512, 256]`, dtype `torch.int32`, stride `[131072, 256, 1]`, device `npu:0`,
  finite and nonzero.

Gate B raw-C2 evidence:

- `official_gmm2_entry_reached: true`.
- `official_gmm2_aic_raw_output_finite: true`.
- `official_gmm2_aic_raw_output_nonzero: true`.
- `official_gmm2_c2v_handoff_verified: true`.
- Strict raw-C2 comparator still fails:
  - `max_abs: 23.6250057220459`
  - `mean_abs: 3.0852577686309814`
  - failed elements over absolute tolerance: `131056` / `131072`
  - NaN/Inf counts: zero for actual, expected, and diff.

Hidden A2 layout variants:

| Variant | Mean abs | Max abs | Failed elements |
|---|---:|---:|---:|
| official high/low halves, low/high nibble order | `3.0852577686309814` | `23.6250057220459` | `131056` |
| high/low halves, high/low nibble order | `3.1224827766418457` | `24.88486099243164` | `131060` |
| swapped low/high halves, low/high nibble order | `7.387302398681641` | `44.44057083129883` | `131069` |
| swapped low/high halves, high/low nibble order | `7.347159385681152` | `44.067378997802734` | `131065` |

W2 layout variants:

| Variant | Mean abs | Max abs | Failed elements | Best-row nonidentity |
|---|---:|---:|---:|---:|
| current row-major host reference | `3.0852577686309814` | `23.6250057220459` | `131056` | `63` |
| CATLASS `nZ` / `weightNz=true` host interpretation | `3.1503801345825195` | `28.72287368774414` | `131058` | `63` |

CATLASS `nZ` diagnostic shape:

- experts: `8`
- K rows: `512`
- packed INT32 columns: `256`
- output columns: `2048`
- flattened INT4 values per expert: `1048576`
- rows round: `512`
- cols round: `2048`
- max offset: `1048575`

Rejected root-cause hypothesis:

- The remaining raw-C2 mismatch is not explained by a simple host W2 physical-layout comparator mismatch.
- Evidence: the current row-major host comparator is still the best W2-layout variant by mean absolute error;
  the CATLASS `nZ` interpretation is slightly worse, and both variants fail almost every element.
- The previous hidden half/nibble layout variants remain consistent: the official high/low, low/high-nibble
  interpretation is still best, while swapped-half variants are much worse.

Current unresolved boundary:

- Gate A is exact and Gate B raw C2 is finite/nonzero, but strict Gate B numerical parity still fails.
- Since simple A2 half/nibble and W2 `weightNz=true` host-layout mismatches are rejected, the next work
  should return to the appendix-required official lifecycle/state table for GMM2 AIC tile state,
  accumulator/D2 mapping, C2V handoff, and `BlockEpilogue2` source mapping before any further behavioral patch.
- Stage 2.3 SVDQ BF16 projections, mixed AIV epilogues, SwiGLU, hidden quantization, final combine, and
  production host tiling remain blocked.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check`: passed.
- Four-visible-NPU raw-C2 probe: completed on `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, wrote summary, and
  correctly reported Stage 2.2 `passed: false`.

## External Hidden Override Pre-Barrier Diagnostic - 2026-06-26T18:34Z

This section is historical evidence. The `2026-06-26T19:00Z` section above supersedes it for current status.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match and post-override readback both report mismatch count 0 after the pre-barrier patch. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | The external debug override handoff is now deterministic, and raw C2 is finite/nonzero, but strict raw-C2 parity against the unfused reference still fails. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 raw-C2 parity. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints remain unchanged:

- Use only `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Do not use or debug public `torch_npu.npu_grouped_matmul`.
- Official `dispatch_ffn_combine_w4_a8` remains the only W4A8 source of truth.
- Do not advance to SVDQ down, final combine, or production enablement while Stage 2.2 fails.

Appendix `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding.
The change below preserves the official full-lifecycle GMM2 path and only fixes the external debug
hidden/scale replacement boundary.

Files changed for this attempt:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - Added `AscendC::SyncAll<true>()` inside `HasExternalGMM2HiddenOverride(params)` before core 0 copies
    external packed hidden and external hidden scale into the official `gmA2I4_I8` and `gmPerTokenScale2`
    buffers.
  - The existing post-copy `SyncAll<true>()` remains in place before the official `SYNCFLAGV2C` handoff
    to GMM2.
  - The ordinary official W4A8 path is unchanged because the new barrier is only entered when an
    external GMM2 hidden override is supplied.

Official source locations used to justify the patch:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp:184-206`:
  `BlockEpilogue1` distributes epilogue rows across AIV cores; each core may return at a different time.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp:388`
  and `:412`: AIV cores write the high and low packed hidden halves into `gmA2I4_I8`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp:422`:
  AIV cores write `gmPerTokenScale2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1326-1330`:
  the official full-lifecycle path calls `BlockEpilogue1`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1332-1345`:
  the debug override now waits for all AIV cores before core 0 replaces packed hidden and scale.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1348-1350`:
  the existing post-copy barrier still precedes the official GMM2 notify flag.

Build/install evidence:

- Focused debug kernel rebuild:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_build_kernel.log`
- `ops_aclnn` rebuild:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_build_ops_aclnn.log`
- `cust_opapi` rebuild:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_build_cust_opapi.log`
- Ascend custom-op config generation:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_ops_config.log`
- Staged package install:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_cmake_install.log`
- Repo-local custom OPP install:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_install_repo_custom_ops.log`,
  result `SUCCESS`.
- System OPP install:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_install_system_opp.log`,
  result `SUCCESS`.
- Installed `libcust_opapi.so` checksum in repo-local and system OPP locations:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_override_prebarrier_installed_opapi_sha.log`,
  SHA256 `8cfa363941ac96c04904cf4c0b727b9e4f17ae4d17de740283333fa6b043d2d5`.

Raw-C2 override pre-barrier probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`,
  repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and
  `--swiglu-limit 454545`.
- Preflight `npu-smi info` showed physical NPUs 0-3 as 910B4 with no running NPU processes.
- NPU preflight log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_override_prebarrier_npu_smi.log`
- Probe log:
  `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_override_prebarrier_top1_expert0_max64.log`
- Summary:
  `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_override_prebarrier_top1_expert0_max64.json`
- Top-level `passed: false`; Stage 2.2 remains failed.
- `production_svdq_host_tiling_fail_closed: true`.
- Source packed hidden exact-match remains zero mismatch.
- Post-override packed hidden readback now matches exactly:
  `hidden_q_post_override_readback_exact_reference.mismatch_count: 0`.
- Post-override hidden scale readback now matches exactly:
  `hidden_scale_post_override_readback_error.exact_mismatch_count: 0`,
  `max_abs: 0.0`.
- `official_gmm2_aic_raw_output_nonzero: true`.
- `official_gmm2_post_dequant_nonzero: false` in the raw-C2 sentinel run.
- Best layout variant by mean absolute error remains `official_high_low_low_high_nibbles`.
- Raw-C2 unfused reference error still fails:
  - `max_abs: 23.6250057220459`
  - `mean_abs: 3.0852577686309814`
- Readback-hidden raw-C2 reference error is identical:
  - `max_abs: 23.6250057220459`
  - `mean_abs: 3.0852577686309814`
- Hidden readback mismatched row count is now `0`.

Conclusion:

- The previous post-override hidden/scale corruption was a real debug-boundary race: core 0 could copy
  external packed hidden and scale before other AIV cores finished the official `BlockEpilogue1` writes.
- The added pre-override AIV barrier fixes that boundary without changing official no-override W4A8
  behavior.
- This does not close Stage 2.2. The current blocker has moved to official-path raw-C2 parity for
  modified hidden after the override data are confirmed exact.
- Stage 2.3 SVDQ BF16 projections, mixed AIV epilogues, SwiGLU, hidden quantization, and production
  host tiling remain blocked.

Validation before this report update:

- Focused kernel build: passed.
- `ops_aclnn` build: passed.
- `cust_opapi` build: passed.
- Repo-local and system OPP installs: passed.
- Four-NPU raw-C2 probe: ran on `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, wrote summary, and correctly
  reported Stage 2.2 `passed: false`.
- `git diff --check`: passed.
- Focused ABI/schema test:
  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`,
  result `1 passed, 16 warnings`.

## Raw C2 Layout Variant Diagnostic - 2026-06-26T18:03Z

This section is historical evidence. The `2026-06-26T18:34Z` section above supersedes it for current status.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match still reports mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Full-lifecycle raw-C2 diagnostic is finite and nonzero, but the strict scaled Gate B comparator still fails. The new layout variant diagnostic rejects a simple high/low half-order or nibble-order host-reference mismatch as the primary explanation. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints remain unchanged:

- Use only `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Do not use or debug public `torch_npu.npu_grouped_matmul`.
- Official `dispatch_ffn_combine_w4_a8` remains the only W4A8 source of truth.
- Do not advance to SVDQ down, final combine, or production enablement while Stage 2.2 fails.

Appendix `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` remains binding. The diagnostic below is a Python-side evidence extension only; it does not modify GMM2 behavior, C2V flags, AIC/AIV role ownership, public grouped matmul, or production host tiling.

Files changed for this attempt:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added `official_a2_c2_physical_layout_contract` to the raw-C2 JSON evidence.
  - Added `raw_c2_layout_variant_diagnostics` for four host-reference interpretations: official high-half/low-half with low/high nibble unpacking, high-half/low-half with high/low nibble unpacking, swapped low/high halves with low/high nibble unpacking, and swapped low/high halves with high/low nibble unpacking.
  - Each variant is explicitly marked diagnostic-only and is not a proposed repack, scale formula, or substitute GMM2 implementation.

Official source locations recorded in the new diagnostic:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp:210-214`: `ChunkTileLen = blockN / 2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp:234`: producer row starts at `gmD[loopIdx * ChunkTileLen]`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp:388` and `:412`: producer writes high-half bytes then low-half bytes.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:309-314`: official `layoutA2{m,k2}` and `layoutD1{maxOutputSize,k2}`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:690-724`: GMM2 sets `n2=k`, `k2=n/2`, then doubles `currentM` for INT4.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:752-767`: GMM2 consumes `gmA2I4` and advances by `M*K` INT4 elements.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:154-199`: `BlockEpilogue2` reads adjacent high/low C2 rows and computes `high * 16 + low`.

Computed official A2/C2 layout for the top-1 expert-0/max64 probe:

- Active rows: `64`.
- Packed hidden row bytes: `512`.
- Producer high-half bytes: `256`; low-half bytes: `256`.
- GMM2 A2 logical rows after INT4 M doubling: `128`.
- GMM2 A2 K columns: `512` INT4 values.
- GMM2 A2 group size: `65536` INT4 elements, equal to `32768` physical bytes.
- Producer D1 group size: `32768` physical bytes.
- A2 physical bytes match producer D1 bytes: `true`.
- GMM2 C2 output columns `n2`: `2048`.
- C2 high/low row offset: `2048`.

Raw-C2 layout variant diagnostic probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`, repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and `--swiglu-limit 454545`.
- Preflight `npu-smi info` showed physical NPUs 0-3 as 910B4 with no running NPU processes.
- Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_layout_variants_top1_expert0_max64.log`
- NPU preflight log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_layout_variants_npu_smi.log`
- Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_layout_variants_top1_expert0_max64.json`
- Top-level `passed: false`; Stage 2.2 remains failed.
- Gate A source packed hidden remains exact: `hidden_packed_exact: true`, mismatch count zero.
- Gate A post-override packed readback still fails in this run: `mismatch_count: 10180`.
- Gate A post-override scale readback still fails in this run: `exact_mismatch_count: 8`.
- Gate B raw C2 is finite and nonzero; C2V readback remains verified for this raw boundary.
- Gate B scaled raw-C2 comparator still fails:
  - `max_abs: 23.6250057220459`
  - `mean_abs: 2.8533122539520264`
  - failed elements: `131053` of `131072`
  - NaN/Inf counts: all zero for actual, expected, and diff.

Layout variant results:

| Variant | Mean abs | Max abs | Failed elements | Best-row nonidentity | Mean best-row abs | Mean diagonal abs |
|---|---:|---:|---:|---:|---:|---:|
| official high/low halves, low/high nibble order | `2.8533122539520264` | `23.6250057220459` | `131053` | `63` | `2.4448091983795166` | `2.8533124923706055` |
| high/low halves, high/low nibble order | `2.8925681114196777` | `24.40673065185547` | `131046` | `63` | `2.3918652534484863` | `2.8925676345825195` |
| swapped low/high halves, low/high nibble order | `7.302852630615234` | `44.44057083129883` | `131069` | `63` | `6.605059623718262` | `7.302852630615234` |
| swapped low/high halves, high/low nibble order | `7.267553329467773` | `44.067378997802734` | `131066` | `63` | `6.502103328704834` | `7.267553329467773` |

Rejected root-cause hypothesis:

- The remaining Gate B mismatch is not explained by a simple host-reference high/low half swap or nibble-order mismatch.
- Evidence: the official expected interpretation is the best variant by mean absolute error, while both half-swapped variants are much worse.
- However, all variants still fail almost every element and retain non-identity best-row alignment for 63 of 64 rows.
- The next unresolved boundary remains official GMM2 physical tile/layout/row-state mapping for modified hidden and the post-override hidden/scale corruption symptom, not a simple packed-byte interpretation change.

Validation before this report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `git diff --check`: passed.

## Raw C2 Parity Diagnostic - 2026-06-26T17:44Z

This section is historical evidence. The `2026-06-26T18:03Z` section above supersedes it for current status.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Accepted prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Source packed hidden exact-match still reports mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Full-lifecycle raw-C2 diagnostic proves the official GMM2/C2V/BlockEpilogue2 raw high-low decode is finite and nonzero, but Gate B scaled raw-C2 reference comparison still fails. Row diagnostics reject hidden-copy corruption as the sole cause, and parity diagnostics reject a simple adjacent even/odd row collapse or high/low row-pair swap as the sole cause. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | No production host-tiling enablement. |

Binding constraints remain unchanged:

- Use only `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Do not use or debug public `torch_npu.npu_grouped_matmul`.
- Official `dispatch_ffn_combine_w4_a8` remains the only W4A8 source of truth.
- Do not advance to SVDQ down, final combine, or production enablement while Stage 2.2 fails.

Appendix `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was re-read on
2026-06-26 and remains binding. It supersedes speculative GMM2-only state-machine edits. No further
behavioral GMM2 patch is permitted until the exact official-vs-debug state deviation it corrects is
recorded here, with source locations for the official producer, consumer, address layout, flag timing,
wait timing, and final drain.

Source locations inspected and used:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h:215`: `InitGMM2OnlyFromPacked` still calls the full official init path and leaves `gmm2OnlyFromPacked_ = false`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:245`: official AIC task runs `GMM1(params)` then `GMM2(params)`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:266`: official AIV task runs `DispatchAndCombine(params)`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:284`: workspace and peer-memory tensors are bound, including `cumsumMM`, `gmA2I4_I8`, `gmC2`, `gmPerTokenScale2`, `tokenPerExpert`, and `preSumBeforeRank`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:684`: official `GMM2(params)` loops experts, waits V2C, calls `blockMmad(...)` on `gmA2I4`, `gmB2`, `gmS2`, and writes `gmC2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1294`: full-lifecycle `BlockEpilogue2` params are built, including the raw-C2 sentinel.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1325`: official hidden producer writes `gmA2I4_I8` and `gmPerTokenScale2`; the debug override copies validated external hidden/scale into the same official region.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1350`: debug readback copies the official post-override `gmA2I4_I8` and `gmPerTokenScale2` regions.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp:1409`: `CombineV2(params, blockEpilogue2)` waits GMM2 flags and invokes `BlockEpilogue2` on `gmCGMM2`, `gmC2`, and `gmPerTokenScale2`.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:143`: `BlockEpilogue2` reads high/low `gmC2`, applies aux bias and per-token hidden scale, writes the W4A8_DEBUG FP32 tap, casts to BF16, and routes peer output.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp:200`: `rawDebugOnly` reads official `gmC2`, decodes high/low FP16 halves into FP32 with `high * 16 + low`, and writes the existing W4A8_DEBUG `gmGMM2` tap before aux bias, hidden scale, BF16 cast, or peer-output routing.

Official-vs-debug state table:

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| packed hidden `gmA2I4_I8` | `BlockEpilogue1` in `DispatchAndCombine` writes packed SwiGLU hidden at `dispatch_ffn_combine_w4_a8_kernel.hpp:1325`. | `GMM2` reads `gmA2I4` at `dispatch_ffn_combine_w4_a8_kernel.hpp:759`. | Bound in `initBuffer` from `workspaceInfo.ptrA2Int4` at `dispatch_ffn_combine_w4_a8_kernel.hpp:293`. | `workspaceInfo.ptrA2Int4`; override uses `gmOffsetD = params.layoutD1.GetOffset(offsetC)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1324`. | Copy length is `curRowNum * (params.problemShape.n() / 2)` bytes; GMM2 uses `layoutA = params.layoutA2.GetTileLayout(...)`. | V2C/C2V cross-core flags around producer and consumer. | Full lifecycle sets `SYNCFLAGV2C` after override at `dispatch_ffn_combine_w4_a8_kernel.hpp:1347`. | GMM2 waits `SYNCFLAGV2C` at `dispatch_ffn_combine_w4_a8_kernel.hpp:737`. | `blockMmad.SynchronizeBlock()` and `Finalize(...)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:775`. | External Stage 2.1 packed hidden is copied into the official region on AIV core 0 at `dispatch_ffn_combine_w4_a8_kernel.hpp:1332`; source packed tensor is exact, but post-override readback is not exact. |
| hidden scale `gmPerTokenScale2` | `BlockEpilogue1` writes per-token scale at `dispatch_ffn_combine_w4_a8_kernel.hpp:1327`. | `BlockEpilogue2` consumes `gmPerTokenScale2` at `block_epilogue_w4a8post_pertoken_v2.hpp:221`. | Bound from `workspaceInfo.ptrPerTokenScale2` at `dispatch_ffn_combine_w4_a8_kernel.hpp:306`. | `workspaceInfo.ptrPerTokenScale2`; override starts at `rowStartThisCore`. | One FP32 scale per active row. | Same full-lifecycle V2C/C2V handoff. | Scale override is before `SYNCFLAGV2C`. | `CombineV2` waits GMM2 flags before epilogue. | `BlockEpilogue2::Finalize()` drains UB events. | External hidden scale is copied into the official region at `dispatch_ffn_combine_w4_a8_kernel.hpp:1338`; post-override scale readback still has exact mismatches. |
| `tokenPerExpert` | Official dispatch/routing and peer exchange populate shmem peer token counts. | `GMM2`, `CombineV2`, and `BlockEpilogue2` read token counts. | Bound from `shmem() + peermemInfo.offsetPeerTokenPerExpert` at `dispatch_ffn_combine_w4_a8_kernel.hpp:309`. | HCCL shmem peer region; layout is `Layout3D(paddedExpertNumAligned, expertPerRank)`. | Expert-major logical rows via `tokenPerExpertLayout`. | Official peer-memory sync and cross-core flags. | Official lifecycle before GMM2/CombineV2. | GMM2/CombineV2 read during expert loops. | `ResetTokenPerExpert(...)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:1375`. | Current debug preserves the full official lifecycle and external `expert_token_nums`; row-identity evidence is still incomplete beyond contiguous top-1 expert-0 tests. |
| `cumsumMM` | Official `GetCumsumForMMAIV`/dispatch state produces cumsum data. | `GMM2` and `CombineV2` use `cumsumMM((EP - 1) * expertPerRank + groupIdx)` for current expert rows. | Bound from `workspaceInfo.ptrcumsumMM` at `dispatch_ffn_combine_w4_a8_kernel.hpp:284`. | `workspaceInfo.ptrcumsumMM`. | One count per expert, accumulated by EP. | Official lifecycle sync. | Ready before GMM2 expert loops. | Read in GMM2 at `dispatch_ffn_combine_w4_a8_kernel.hpp:705` and CombineV2 at `dispatch_ffn_combine_w4_a8_kernel.hpp:1428`. | No custom drain; workspace is reused per launch. | Full lifecycle owns this state. Earlier standalone seeding helpers remain inactive because `gmm2OnlyFromPacked_ = false`. |
| `preSumBeforeRank` | Official dispatch/combine state writes per-rank expert offsets. | `BlockEpilogue2` uses it for peer-output routing at `block_epilogue_w4a8post_pertoken_v2.hpp:252`. | Bound from `workspaceInfo.ptrSumBeforeRank` at `dispatch_ffn_combine_w4_a8_kernel.hpp:314`. | `workspaceInfo.ptrSumBeforeRank`. | EP by expertPerRank. | Official shmem/peer sync. | Before final routing in `BlockEpilogue2`. | Consumed during `BlockEpilogue2` peer-output loop. | Official shmem completion after `CombineV2`. | Full lifecycle owns this state. Standalone zeroing is inactive in the current full-lifecycle path. |
| GMM2 AIC input tile state | Official GMM2 forms `inGroupProblemShape`, `layoutA`, `layoutB2`, `layoutScale`, and `layoutC`. | `blockMmad(...)` consumes the tile state. | Inside `GMM2` expert loop at `dispatch_ffn_combine_w4_a8_kernel.hpp:724`. | A tile starts at `gmGroupOffsetA + gmOffsetA`, `gmGroupOffsetB + gmOffsetB`, `gmOffsetS`, and writes `gmGroupOffsetC + gmOffsetC`. | `L1TileShape<128,256,1024>` with `currentM` doubled for int4. | Waits V2C before processing each sync group. | GMM2 starts after hidden producer signal. | `CrossCoreWaitFlag<0x2>(SYNCFLAGV2C)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:737`. | `blockMmad.SynchronizeBlock()` then `Finalize`. | Current debug uses official tile state. Raw C2 is finite and nonzero, but the scaled official-contract raw-C2 comparator fails almost all elements. |
| GMM2 accumulator / D2 region | `blockMmad` writes `gmC2` at `dispatch_ffn_combine_w4_a8_kernel.hpp:759`. | `BlockEpilogue2` reads `gmC2` high/low halves. | Bound from `workspaceInfo.ptrC2` at `dispatch_ffn_combine_w4_a8_kernel.hpp:302`. | `workspaceInfo.ptrC2`; `BlockEpilogue2` computes high/low offsets from `preSrcExpertSum`, `blockCoord`, and `params.n2`. | High and low halves are separated by `params.n2`. | GMM2 completion flags consumed by CombineV2. | GMM2 signals through blockMmad/flag lifecycle. | CombineV2 waits flags at `dispatch_ffn_combine_w4_a8_kernel.hpp:1460`. | `blockMmad.Finalize(...)`. | Raw-C2 debug proves this boundary is finite and nonzero; the strict scaled raw-C2 comparator is implemented and fails, so Gate B remains failed. |
| C2V handoff state | Official GMM2 producer and CATLASS async MMAD lifecycle. | `CombineV2` and `BlockEpilogue2`. | Inside `GMM2` and `CombineV2`. | Cross-core flags, not a tensor region. | Per expert sync-loop. | `SYNCFLAGV2C` and per-loop flags. | After hidden producer and after GMM2 tile completion. | `CombineV2` waits flags before each epilogue tile. | Final wait loop at `dispatch_ffn_combine_w4_a8_kernel.hpp:1482`. | Raw-C2 readback implies a working C2V path for the tested shape, but full numerical gate still fails. |
| `BlockEpilogue2` input state | `CombineV2` supplies `gmCGMM2`, `gmC2`, `gmPerTokenScale2`, aux bias, tile coord/shape, group id, and prefix state. | `BlockEpilogue2::operator()` consumes them. | Params built at `dispatch_ffn_combine_w4_a8_kernel.hpp:1294`; flags initialized at `block_epilogue_w4a8post_pertoken_v2.hpp:117`. | Reads `gmC2`, writes debug `gmCGMM2`, and routes to peer output using `offsetD`. | AIV slices M in 32-row chunks. | UB/MTE flags plus W4A8_DEBUG `EVENT_ID7`. | `InitFlag()` before loop. | Per-tile MTE waits in `BlockEpilogue2`. | `Finalize()` waits outstanding UB events. | Current debug uses the official epilogue and only adds raw-debug early return under sentinel `450000 < swigluLimit < 460000`. |
| FP32 post-dequant debug tap | `BlockEpilogue2` writes W4A8_DEBUG `gmGMM2` after aux and hidden scale at `block_epilogue_w4a8post_pertoken_v2.hpp:231`. | Python probe reads `gmm2_post_dequant`. | Debug pointer passed through `Init(... debugGMM2GM)` and `Params::ptrDebugGMM2`. | `workspaceInfo.ptrCGMM2` when debug enabled; tap offset is `(preSrcExpertSum * n2 + blockCoord.m() * n2) / 2 + blockCoord.n()`. | FP32 row-major active rows by hidden size. | W4A8_DEBUG `EVENT_ID7`. | After high/low decode, aux add, and hidden-scale multiply. | Readback after kernel completion. | `BlockEpilogue2::Finalize()`. | Normal post-dequant is finite/nonzero but fails the strict unfused reference; Gate C remains failed. |

Files changed for this attempt:

- `tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py`
  - Added `_official_gmm2_raw_c2_reference(...)` as a Gate B comparator for the official scaled C2 boundary. This is diagnostic evidence only, not an implementation path.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - Added `IsFullLifecycleGMM2RawDebug(params)`, enabled only for `450000.0 < swigluLimit < 460000.0`.
  - Passes that sentinel into the existing `BlockEpilogue2::Params rawDebugOnly` flag in the full official lifecycle.
- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
  - Added parsing/reporting for the raw-C2 diagnostic mode.
  - Added strict predeclared raw-C2 comparator tolerances, failed-element counts, relative error, and NaN/Inf counts.
  - Added raw-C2 row diagnostics comparing source hidden, post-override readback hidden, exact hidden rows, corrupted hidden rows, and best row alignment.
  - Added raw-C2 even/odd and adjacent-row pattern diagnostics for the official doubled-M INT4 row interpretation.
  - No Torch schema, adapter ABI, production SVDQ, or public grouped-matmul behavior changed.
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
  - Added the mandatory official-vs-debug state table before any further behavioral GMM2 patch.

No kernel rebuild was required for the 17:20Z comparator update because only Python probe/report code changed after the already-installed raw-C2 debug kernel.

Raw-C2 parity diagnostic probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`, repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and `--swiglu-limit 454545`.
- Preflight: `npu-smi info` showed no active NPU processes on physical NPUs 0-3; Python saw logical `npu_device_count: 4`, selected logical device `0`, runtime SOC `224`.
- Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_parity_top1_expert0_max64.log`
- Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_parity_top1_expert0_max64.json`
- Top-level `passed: false`; Stage 2.2 remains failed.
- Gate A source packed hidden remains exact: `exact_match: true`, `mismatch_count: 0`.
- Gate A post-override packed readback still fails in this run: `mismatch_count: 3290`, first rows `[13, 21, 23, 31, 33, 35, 37, 39]`.
- Gate A post-override scale readback still fails in this run: `exact_mismatch_count: 4`.
- Source-hidden raw-C2 even/odd split:
  - even rows: `row_count: 32`, `mean_abs: 3.14512300491333`, `max_abs: 21.408472061157227`
  - odd rows: `row_count: 32`, `mean_abs: 2.848015785217285`, `max_abs: 23.6250057220459`
  - first adjacent actual even/odd mean-abs values: `[2.902888536453247, 3.2883198261260986, 2.9655027389526367, 2.751413345336914]`
  - first adjacent expected even/odd mean-abs values: `[2.830193519592285, 3.3786818981170654, 2.761028528213501, 3.000396251678467]`
- Readback-hidden raw-C2 even/odd split:
  - even rows: `row_count: 32`, `mean_abs: 3.14512300491333`, `max_abs: 21.408472061157227`
  - odd rows: `row_count: 32`, `mean_abs: 2.5995981693267822`, `max_abs: 23.6250057220459`
- Row-alignment diagnostic remains non-identity:
  - Source-hidden reference: `nonidentity_best_row_count: 63`, `mean_best_row_abs: 2.613636016845703`, `mean_diagonal_abs: 2.9965696334838867`.
  - Readback-hidden reference: `nonidentity_best_row_count: 63`, `mean_best_row_abs: 2.1697492599487305`, `mean_diagonal_abs: 2.8723604679107666`.

Rejected root-cause hypothesis:

- The remaining Gate B mismatch is not explained by a simple adjacent even/odd row collapse or high/low row-pair swap.
- Evidence: actual adjacent even/odd rows differ substantially, and both even and odd logical rows have large raw-C2 comparator error.
- The best-row diagnostic is still non-identity for 63 of 64 rows, so the unresolved boundary remains the official GMM2 physical tile/layout/row-state mapping for modified hidden, not a single parity-only remap.

Raw-C2 row diagnostic probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`, repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and `--swiglu-limit 454545`.
- Preflight: `npu-smi info` showed no active NPU processes on physical NPUs 0-3; Python saw logical `npu_device_count: 4`, selected logical device `0`, runtime SOC `224`.
- Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_rowdiag_top1_expert0_max64.log`
- Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_rowdiag_top1_expert0_max64.json`
- Top-level `passed: false`; Stage 2.2 remains failed.
- Gate A source packed hidden remains exact: `exact_match: true`, `mismatch_count: 0`.
- Gate A post-override packed readback still fails: `mismatch_count: 10816`, first rows `[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 34, 35, 37, 39, 45, 47, 59, 61, 63]`.
- Gate A post-override scale readback still fails in this run: `exact_mismatch_count: 22`.
- Gate B source-hidden scaled raw-C2 comparator:
  - `passed: false`
  - `max_abs: 23.6250057220459`
  - `mean_abs: 2.808938980102539`
  - `failed_element_count_abs_gt_tolerance: 131052` of `131072`
- Gate B post-override-readback-hidden scaled raw-C2 comparator:
  - `passed: false`
  - `max_abs: 23.6250057220459`
  - `mean_abs: 2.397221803665161`
  - `failed_element_count_abs_gt_tolerance: 130842` of `131072`
- Row split:
  - hidden readback exact rows: `39`
  - hidden readback mismatched rows: `25`
  - raw-C2 source-reference error on hidden-exact rows: `max_abs: 23.6250057220459`, `mean_abs: 3.1165595054626465`
  - raw-C2 source-reference error on hidden-mismatched rows: `max_abs: 18.12263298034668`, `mean_abs: 2.3290510177612305`
- Row-alignment diagnostic:
  - Source-hidden reference: `nonidentity_best_row_count: 63`, `mean_best_row_abs: 2.387960195541382`, `mean_diagonal_abs: 2.8089394569396973`.
  - Readback-hidden reference: `nonidentity_best_row_count: 63`, `mean_best_row_abs: 1.872459888458252`, `mean_diagonal_abs: 2.397221565246582`.

Rejected root-cause hypothesis:

- The remaining Gate B mismatch is not explained solely by corrupted post-override hidden readback.
- Evidence: the 39 rows whose post-override hidden readback exactly equals the intended source hidden still fail the source-hidden raw-C2 comparator, and their mean raw-C2 error is larger than the rows with corrupted hidden readback.
- The readback-hidden comparator is closer than the source-hidden comparator but still fails almost all elements, so hidden readback corruption is a contributor or symptom, not a sufficient root cause.
- The next unresolved boundary is official GMM2 tile/layout/row-state semantics for the modified-hidden full lifecycle, including whether the external mixed-hidden rows are mapped to the same physical A2/C2 tile coordinates consumed by AIC GMM2 and AIV `BlockEpilogue2`.

Scaled raw-C2 reference probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`, repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and `--swiglu-limit 454545`.
- Preflight: `npu-smi info` showed physical NPUs 0-3 as 910B4, no active NPU processes; Python saw logical `npu_device_count: 4`, selected logical device `0`, runtime SOC `Ascend910B4`.
- Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_scaled_reference_top1_expert0_max64.log`
- Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_scaled_reference_top1_expert0_max64.json`
- Top-level `passed: false`; Stage 2.2 remains failed.
- Gate A source packed hidden remains exact: `exact_match: true`, `mismatch_count: 0`.
- Gate A post-override packed readback still fails: `mismatch_count: 5861`, first rows `[3, 7, 9, 13, 17, 21, 25, 29, 31, 33, 35, 37, 39]`.
- Gate A post-override scale readback was exact in this run: `exact_mismatch_count: 0`.
- Gate B raw C2 high/low decode:
  - finite: `true`
  - nonzero: `true`
  - shape: `[64, 2048]`
  - `max_abs: 19.904296875`
  - `mean_abs: 1.9497606754302979`
  - `nan_count: 0`
  - `inf_count: 0`
  - first sample: `[-3.12115478515625, 0.0625152587890625, -0.81787109375, 1.11474609375]`
- Gate B scaled raw-C2 comparator:
  - contract: `(high_acc * 16 + low_acc) * postloaded_weight_scale` before scale bias, hidden scale, BF16 cast, or peer-output routing.
  - `passed: false`
  - tolerance: `max_abs <= 0.0002`, `mean_abs <= 0.00002`
  - `max_abs: 23.6250057220459`
  - `mean_abs: 2.939976215362549`
  - `failed_element_count_abs_gt_tolerance: 131056` of `131072`
  - `actual_nan_count: 0`, `actual_inf_count: 0`, `expected_nan_count: 0`, `expected_inf_count: 0`
  - first actual sample: `[-3.12115478515625, 0.0625152587890625, -0.81787109375, 1.11474609375]`
  - first expected sample: `[2.201096534729004, -0.1250970959663391, 1.165556788444519, 1.6457070112228394]`
- Status-field interpretation:
  - `official_gmm2_entry_reached: true`
  - `official_gmm2_aic_raw_output_finite: true`
  - `official_gmm2_aic_raw_output_nonzero: true`
  - `official_gmm2_aic_reference_passed: false`
  - `official_gmm2_c2v_handoff_verified: true`
  - `official_gmm2_post_dequant_reference_passed: false`
  - `official_gmm2_numerical_gate_passed: false`

Rejected comparator hypothesis:

- An intermediate unscaled raw-C2 comparator, saved in `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_reference_top1_expert0_max64.json`, compared raw C2 against integer `high_acc * 16 + low_acc`.
- That was rejected because the official GMM2 call passes `gmS2` into `blockMmad(...)` at `dispatch_ffn_combine_w4_a8_kernel.hpp:759`, while `BlockEpilogue2` only adds aux bias and hidden scale later.
- The corrected Gate B comparator therefore includes postloaded W2 scale, matching the official lifecycle boundary being tapped.

Build/install evidence:

- Kernel rebuild: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_rebuild_kernel.log`
- Packaged install: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_cmake_install_custom_ops.log`
- Repo-local OPP install: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_install_repo_custom_ops.log`, result `SUCCESS`
- System OPP install: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_install_system_opp.log`, result `SUCCESS`
- `libcust_opapi.so` SHA in build, repo-local OPP, and system OPP: `8cfa363941ac96c04904cf4c0b727b9e4f17ae4d17de740283333fa6b043d2d5`
- `libcust_opmaster_rt2.0.so` SHA in build, repo-local OPP, and system OPP: `3e61b3d074c42bbec446e6cbf159a2af36646af138985cc55bdd938e5f6050e6`

Raw C2 diagnostic probe:

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`, repo-local `libcust_opapi.so`, top-1 expert 0, 64 tokens, `max_output_size=64`, and `--swiglu-limit 454545`.
- Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_raw_c2_top1_expert0_max64.log`
- Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_raw_c2_top1_expert0_max64.json`
- Top-level `passed: false` by design because Stage 2.2 is not complete.
- Source packed hidden remains exact: `exact_match: true`, `mismatch_count: 0`.
- Post-override packed readback still fails: `mismatch_count: 2743`, first rows `[27, 29, 31, 33, 35, 37, 39]`.
- Post-override scale readback still fails: `exact_mismatch_count: 2`, indices `[34, 35]`.
- Gate B raw C2 high/low decode:
  - finite: `true`
  - nonzero: `true`
  - `max_abs: 19.904296875`
  - `mean_abs: 2.100557327270508`
  - `nan_count: 0`
  - `inf_count: 0`
  - first sample: `[-3.12115478515625, 0.0625152587890625, -0.81787109375, 1.11474609375]`
- Status-field interpretation:
  - `official_gmm2_entry_reached: true`
  - `official_gmm2_aic_raw_output_finite: true`
  - `official_gmm2_aic_raw_output_nonzero: true`
  - `official_gmm2_c2v_handoff_verified: true` for the raw-C2 readback boundary
  - `official_gmm2_aic_reference_passed: false` because an accumulator-level reference comparison is not implemented yet
  - `official_gmm2_post_dequant_reference_passed: false`
  - `official_gmm2_numerical_gate_passed: false`

Normal post-dequant regression check after the raw-C2 patch:

- Command used the same top-1 expert-0/max64 shape without the raw sentinel.
- Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_post_dequant_top1_expert0_max64_after_raw_c2_patch.log`
- Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_post_dequant_top1_expert0_max64_after_raw_c2_patch.json`
- Top-level `passed: false`.
- Source packed hidden remains exact: `exact_match: true`, `mismatch_count: 0`.
- Post-override packed readback still fails: `mismatch_count: 5992`, first rows `[9, 13, 17, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 47]`.
- Post-override scale readback still fails: `exact_mismatch_count: 12`, indices `[28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]`.
- Post-dequant active output is finite and nonzero: `max_abs: 0.46528252959251404`, `mean_abs: 0.03993524983525276`, `nan_count: 0`, `inf_count: 0`.
- Unfused post-dequant reference still fails: `max_abs: 0.552257776260376`, `mean_abs: 0.06802959740161896`.

Current interpretation:

- The official GMM2 AIC/C2V/BlockEpilogue2 raw decode boundary is alive and nonzero in the full official lifecycle.
- The previous all-zero/raw-unproven hypothesis is superseded.
- Stage 2.2 remains failed because Gate A after official override/readback is still not exact and Gate C post-dequant still fails the strict reference.
- Next work should compare the raw-C2 diagnostic against an official accumulator-level reference or add a tile-local raw comparison, then isolate whether the remaining final mismatch is caused by hidden/scale override corruption, aux/scale dequant, row routing, or post-dequant readback.

Validation before commit:

- `git diff --check`: passed.
- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`: `1 passed, 16 warnings`.

## Rebuild Handoff - 2026-06-26T16:25Z

This section is the latest handoff before the environment rebuild. It supersedes the `2026-06-26T15:58Z` section for current state, but the older section remains valid historical evidence.

Status: Stage 2.2 remains FAIL / IN PROGRESS. Do not proceed to Stage 2.3, SVDQ down composition, final combine, or production host tiling.

Requirements re-read for this handoff:

- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_prompt.md`
- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_prompt.md`
- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_report_clarified_completed.md` was requested by older notes but does not exist; `find /root/workspace/lza -maxdepth 3 -name '*clarified*' -o -name '*completed*'` returned no file.

Binding constraints that still govern the next session:

- Only use logical/physical NPUs `0,1,2,3`; real-device commands must set `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Do not use, modify, reinterpret, or further debug public `torch_npu.npu_grouped_matmul`.
- Official `dispatch_ffn_combine_w4_a8` remains the only source of truth for W4A8 GMM2 packed-weight access, AIC accumulation, AIV dequantization, and debug readback.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.
- Source inspection, manifests, scalar substitutes, public grouped-matmul experiments, or scale-formula guessing are not progress toward the Stage 2.2 gate.

Current official-path behavior:

- `InitGMM2OnlyFromPacked` still runs the full official `DispatchAndCombine` lifecycle and only injects external hidden/scale in the official path.
- The hidden override copy is in `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp` at the `CopyGMToGM(gmA2I4_I8[gmOffsetD], externalHiddenX[gmOffsetD], ...)` and `CopyGMToGM(gmPerTokenScale2[rowStartThisCore], externalHiddenScale[rowStartThisCore], ...)` boundary.
- The readback copy is immediately after the official hidden override path and copies `gmA2I4_I8` to `ptrDebugHiddenX` and `gmPerTokenScale2` to `ptrDebugHiddenScale`.
- For `expertPerRank=8`, top-1 expert-0 routing gives `dequantSum` equivalent to `[0, 64, 64, 64, 64]`; rows 0-63 should be covered in one chunk. The remaining mismatch is therefore not explained only by padded capacity.

Latest constrained `max_output_size=64` probe evidence:

| Probe | Summary | Packed hidden source | Post-override packed readback | Post-override scale readback | GMM2 reference |
|---|---|---|---|---|---|
| Baseline source before workspace experiment | `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_hidden_readback_top1_expert0_max64.json` | exact, mismatch count 0 | fail, 4057 mismatches across rows `[19, 25, 27, 29, 31, 33, 35, 37, 39]` | fail, 4 mismatches at indices `[36, 37, 38, 39]` | fail, max abs `0.552257776260376`, mean abs `0.07091927528381348` |
| Guarded debug-workspace host-tiling experiment | `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_hidden_readback_top1_expert0_max64_after_workspace_fix.json` | exact, mismatch count 0 | fail, 6760 mismatches across rows `[5, 11, 15, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 45]` | fail, 10 mismatches at indices `[28, 29, 30, 31, 32, 33, 36, 37, 38, 39]` | fail, max abs `0.552257776260376`, mean abs `0.06795954704284668` |
| Unguarded debug-workspace host-tiling experiment | `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_hidden_readback_top1_expert0_max64_after_workspace_fix_unguarded.json` | exact, mismatch count 0 | fail, 19721 mismatches across 44 rows; first rows `[3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]` | fail, 39 mismatches; first indices `[2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]` | fail, max abs `0.552257776260376`, mean abs `0.0572160966694355` |

Workspace experiment result:

- Hypothesis tested: the GMM2-only debug op may under-declare workspace because `WorkspaceInfo` can allocate internal FP32 debug taps for GMM1/GMM1-hidden/GMM2 when some external debug pointers are null.
- Guarded source patch built and installed but likely did not affect the shared host tiling object because the tiling object is not compiled with the same `W4A8_DEBUG` definition.
- Unguarded source patch produced a new host library SHA `ff02c29715eab1a31ce7fe6037fe5ab77f48ff8a671efc0b1c52a9b43025a4eb` and installed it into both repo-local and system OPP locations, but the probe got worse.
- Because the unguarded change did not help and grew production W4A8 workspace, the source patch was reverted before commit. There is no retained production-facing code change from this experiment.
- Important environment note: the currently installed repo-local and system OPP host libraries were overwritten by the unguarded experiment before the source revert. A fresh environment should rebuild/install from committed source; do not treat the current installed OPP as the committed source state.

Build/install logs for the negative workspace experiment:

- Guarded rebuild/install:
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_rebuild_ops_aclnn.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_rebuild_cust_opapi.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_cmake_install_custom_ops.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_install_repo_custom_ops.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_install_system_opp.log`
- Unguarded rebuild/install:
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_rebuild_cust_opmaster_unguarded.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_unguarded_cmake_install_custom_ops.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_unguarded_install_repo_custom_ops.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_workspace_debug_unguarded_install_system_opp.log`
  - `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_hidden_readback_top1_expert0_max64_after_workspace_fix_unguarded.log`

Next required work:

- Keep Stage 2.2 focused on the official hidden/scale override and readback boundary.
- Do not continue the workspace-size hypothesis unless new evidence directly proves workspace overlap.
- Add a decisive official-path diagnostic for hidden override timing, row mapping, chunk/tail behavior, and C2V/Gate B raw AIC output.
- Gate A remains valid only at the source hidden packed tensor. Gate A after official override/readback still fails. Gate B raw AIC remains unproven. Gate C post-dequant remains nonzero but reference-failing.

Validation before handoff commit:

- `git diff --check`: passed.
- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`: `1 passed, 16 warnings`.

## Authoritative Status - 2026-06-26T15:58Z

This table supersedes ambiguous status statements in older handoff sections. Older sections are historical evidence unless explicitly referenced by the latest active section.

| Item | Status | Evidence / blocker |
|---|---|---|
| Stage 2.0 seven-output mixed epilogue debug ABI | PASS | Built, installed, registered, and launched on logical NPU 0 in prior Stage 2 evidence. |
| Stage 2.1 canonical hidden INT8 / packed INT4 boundary | PASS | Packed hidden exact-match gate passed with mismatch count 0. |
| Stage 2.2 modified-hidden official W4A8 GMM2 | FAIL / IN PROGRESS | Tuple Torch ABI, generated opapi, repo-local OPP, and system OPP are now refreshed. The official GMM2 debug op launches and returns finite nonzero post-dequant output, but post-override packed hidden/scale readback is not exact and the official-contract unfused GMM2 reference still fails. |
| Stage 2.3 and later | BLOCKED | Blocked on Stage 2.2 Gate B/Gate C. |
| Production `DispatchFFNCombineW4A8SVDQ` | FAIL-CLOSED | Host tiling must remain fail-closed until Stage 2.2+ production gates pass. |

## Current State - 2026-06-26T15:58Z

Status: Stage 2.2 remains FAIL / IN PROGRESS.

Binding constraints for the next environment:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` remains the active working prompt.
- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was read and is binding.
- `svdq_moe_clean_rewrite_reference.md` and all Stage 2 appendices remain background requirements.
- Do not modify, reinterpret, further debug, or use public `torch_npu.npu_grouped_matmul`.
- The official `dispatch_ffn_combine_w4_a8` implementation remains the only W4A8 GMM2 source of truth.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.
- Real-device commands must keep `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.

### Runtime ABI and OpAPI Refresh - 2026-06-26T15:58Z

Purpose:

- Make the installed runtime match the source-level three-output `SVDQW4A8GMM2DebugReadback` ABI.
- Keep the diagnostic isolated to the official `dispatch_ffn_combine_w4_a8` W4A8 GMM2 path.
- Do not change production SVDQ host tiling.
- Do not use public grouped matmul, scalar substitutes, or scale-formula guessing as progress.

Runtime issue found and fixed:

- The source tree already declared `svdq_w4a8_gmm2_debug_readback(...) -> (Tensor gmm2_post_dequant, Tensor hidden_x_readback, Tensor hidden_scale_readback)`, but the installed Python extension was stale and still exposed a single-output schema.
- Before refresh, `_dispatch_dump` evidence is in `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_before_extension_rebuild.log`.
- Rebuilt and installed `vllm_ascend_C`:
  - Build log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_rebuild_vllm_ascend_C.log`
  - Install log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_install_vllm_ascend_C.log`
  - After-refresh schema log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_after_extension_rebuild.log`
- After extension refresh, the first real-device probe failed with `executor == nullptr` because `libcust_opapi.so` was also stale and only registered `gmm2PostDequantOut`.
- Rebuilt and installed opapi/custom OPP:
  - `ops_aclnn` rebuild log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_rebuild_ops_aclnn.log`
  - `cust_opapi` rebuild log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_rebuild_cust_opapi.log`
  - Packaged install log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_cmake_install_custom_ops.log`
  - Repo-local OPP install log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_install_repo_custom_ops.log`, result `SUCCESS`
  - System OPP install log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_install_system_opp.log`, result `SUCCESS`
- The refreshed `libcust_opapi.so` SHA is `8cfa363941ac96c04904cf4c0b727b9e4f17ae4d17de740283333fa6b043d2d5` in all checked locations:
  - `csrc/build/libcust_opapi.so`
  - `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so`
  - `/usr/local/Ascend/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so`

Validation evidence:

- ABI/source registration test after extension refresh:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`
  - Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_abi_after_extension_rebuild.log`
  - Result: `1 passed, 16 warnings`.
- NPU visibility check: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_tuple_schema_npu_smi.log`; only NPUs 0-3 were visible/used.

### Latest Hidden-Boundary Readback Probe - 2026-06-26T15:58Z

The latest probe is a real-device official GMM2 debug-op launch after both tuple ABI and opapi refreshes.

- Command used `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, repo-local `ASCEND_CUSTOM_OPP_PATH`, and repo-local `libcust_opapi.so` on `LD_LIBRARY_PATH`.
- Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_hidden_readback_top1_expert0_after_tuple_opapi_mismatch_detail.log`
- Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_hidden_readback_top1_expert0_after_tuple_opapi_mismatch_detail.json`
- Top-level: `passed: false`, `preflight_failed: false`, `torch_op_registered: true`, `production_svdq_host_tiling_fail_closed: true`.
- Environment: `npu_device_count: 4`, `selected_device: 0`, `runtime_soc_version: Ascend910B4`.
- Shape: `num_tokens: 64`, `top_k: 1`, `active_rows: 64`, `hidden_size: 2048`, `intermediate_size: 512`, `local_num_experts: 8`.
- Routing: `expert_token_nums: [[64, 0, 0, 0, 0, 0, 0, 0]]`, `expert_token_total: 64`, `expert_contiguous_rows: true`.

Gate A source boundary:

- `canonical_hidden_bf16`: finite and nonzero.
- `hidden_int8`: finite and nonzero.
- `hidden_int4_packed`: finite and nonzero.
- `hidden_scale`: finite and nonzero.
- `hidden_q_packed_exact_reference.exact_match: true`.
- `hidden_q_packed_exact_reference.mismatch_count: 0`.
- This proves the deterministic mixed-epilogue packed-INT4 calibration source remains valid.

Post-override hidden/scale readback:

- `hidden_q_post_override_readback_exact_reference.exact_match: false`.
- `hidden_q_post_override_readback_exact_reference.mismatch_count: 7047`.
- `hidden_q_post_override_readback_exact_reference.mismatch_row_count: 16`.
- First mismatching rows: `[5, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 43]`.
- `hidden_q_post_override_readback_exact_reference.mismatch_col_count: 512`.
- First mismatch example: row `5`, col `32`, actual `0`, expected `1`.
- `hidden_scale_post_override_readback_error.exact_mismatch_count: 20`.
- Scale mismatch indices: `[16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]`.
- First scale mismatch example: index `16`, actual `0.0`, expected `0.026451772078871727`.
- `hidden_scale_post_override_readback_error.max_abs: 0.03198818862438202`.
- `hidden_scale_post_override_readback_error.mean_abs: 0.007841335609555244`.
- `hidden_post_override_readback_exact: false`.
- `hidden_scale_post_override_readback_exact: false`.

Gate C post-dequant:

- `official_gmm2_kernel_launched: true`.
- `gmm2.post_dequant_active.finite: true`.
- `gmm2.post_dequant_active.nonzero: true`.
- `gmm2.post_dequant_active.max_abs: 0.46528252959251404`.
- `gmm2.post_dequant_active.mean_abs: 0.03387366980314255`.
- Actual sample begins `[-0.0737280622124672, 0.0014767383690923452, -0.019319789484143257, 0.026332585141062737]`.
- `gmm2.unfused_reference.passed: false`.
- Reference error: `max_abs: 0.552257776260376`, `mean_abs: 0.06560951471328735`.
- Expected sample begins `[0.051994405686855316, -0.0029550495091825724, 0.02753283828496933, 0.03887496888637543]`.

Required status fields:

- `official_gmm2_entry_reached`: true by the successful official debug-op launch.
- `official_gmm2_loop_count`: not measured in this non-loop-stats readback run.
- `official_gmm2_active_tile_count`: not measured in this non-loop-stats readback run.
- `official_gmm2_aic_raw_output_finite`: unproven.
- `official_gmm2_aic_raw_output_nonzero`: unproven.
- `official_gmm2_aic_reference_passed`: false / unproven.
- `official_gmm2_c2v_handoff_verified`: unproven.
- `official_gmm2_post_dequant_finite`: true.
- `official_gmm2_post_dequant_nonzero`: true.
- `official_gmm2_post_dequant_reference_passed`: false.
- `official_gmm2_numerical_gate_passed`: false.

| State or region | Official producer / consumer | Current debug behavior after this attempt | Status |
|---|---|---|---|
| Packed hidden `gmA2I4_I8` | Official AIV `BlockEpilogue1` produces packed hidden rows, then official AIC `GMM2(params)` consumes the same GM boundary. | `InitGMM2OnlyFromPacked` passes `debugHiddenXGM` into the official debug `Init(...)` path, and the probe now returns a non-null post-override readback tensor. | Source packed hidden is exact, but post-override readback is not exact. Next work must inspect the official hidden override/copy boundary, row mapping, and chunking. |
| Hidden scale `gmPerTokenScale2` | Official AIV `BlockEpilogue1` writes one FP32 scale per hidden row, then GMM2/dequant consume it. | `InitGMM2OnlyFromPacked` passes `debugHiddenScaleGM` into the official debug init path, and the probe now returns a non-null post-override scale readback tensor. | Scale is finite/nonzero at source, but post-override readback contains zeroed rows. Next work must fix this before treating GMM2 numerics as a valid Gate C signal. |
| FP32 post-dequant debug tap | Official `BlockEpilogue2` / `CombineV2` writes the W4A8_DEBUG FP32 GMM2 post-dequant tap. | Existing `gmm2PostDequant` output is unchanged. | It is now finite and nonzero, proving the stale ABI/opapi blocker is gone, but it does not match the official-contract unfused reference. |

Interpretation:

- The stale Python extension and stale `libcust_opapi.so` blockers are fixed.
- The deterministic mixed-epilogue packed hidden source is exact.
- The official GMM2 debug kernel launches and no longer returns all-zero post-dequant output.
- Stage 2.2 still cannot pass because the post-override hidden packed/scale boundary is not exact and the post-dequant output fails the official-contract reference.
- The next session should debug the official-path hidden/scale override timing, row mapping, and chunk/tail behavior using the newly recorded mismatch rows. Do not proceed to Stage 2.3, SVDQ down composition, final combine, or production host tiling.

Files changed by the latest diagnostic/report update:

- `tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`
- `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
- `/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_report.md`

Validation run for latest diagnostic/report update:

- `python -m py_compile tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py`: passed.
- Real-device probe above completed and wrote the summary; process exit was nonzero because `passed: false`.
- `git diff --check`: passed.

## Historical State - 2026-06-26T13:45Z

This section is superseded by the `2026-06-26T15:58Z` current state above. It is retained as failed-experiment evidence.

## Current State - 2026-06-26T13:45Z

Status: Stage 2.2 remains FAIL / IN PROGRESS.

Binding prompt and appendix state:

- `svdq_qwen35_moe_clean_implementation_stage2_prompt.md` remains the active working prompt.
- `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was read and is binding for this handoff.
- The public `torch_npu.npu_grouped_matmul` path was not modified, debugged, or used as progress.
- The official `dispatch_ffn_combine_w4_a8` producer/dequant path remains the only source of truth for W4A8 GMM2.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed.

### Official vs Debug GMM2 State Table - 2026-06-26T12:39Z

This table is the required pre-patch state comparison from `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`. It documents the official full-W4A8 state lifecycle before any further isolated GMM2-only behavior change.

| State boundary | Official full-W4A8 path | Current isolated GMM2-only debug path | Status / required action |
|---|---|---|---|
| Source of token counts | `moe_init_routing_quant_v2` writes the local rank's count matrix into peer memory at `localTokenPerExpert = shmem() + offsetPeerTokenPerExpert + tokenPerExpertLayout(rank, 0, 0) * sizeof(int32_t)`. | Host probe passes deterministic rank-2 INT32 `external_expert_token_nums` with shape `[1, experts]` and nonzero routed expert counts. | Host boundary is valid; no public grouped-matmul or alternative count path is involved. |
| Peer token matrix layout | `Layout3D(dim0, dim1, dim2) = dim0 * paddedExpertNumAligned + dim1 * expertPerRank + dim2`. Official peer exchange materializes the padded `EP x EP x expertPerRank` token matrix under `tokenPerExpert`. | `SeedGMM2OnlyTokenState` currently copies external counts only to `tokenPerExpert[tokenPerExpertLayout(rank, 0, 0)]`. | This is compatible with official layout addressing for `tokenPerExpert(dstEpIdx, rank, groupIdx)` but is not sufficient by itself for the current cumsum reader. |
| Cumsum producer | After peer sync, official AIV core 0 calls `GetCumsumForMMAIV(tokenPerExpert, cumsumMM, expertPerRank, rank, EP)`. That helper reads from `tokenPerExpert[rank * expertPerRank]` with padded row stride. | GMM2-only AIC and AIV both call `SeedGMM2OnlyTokenState`; loop-stats proves `cumsumMM[(EP - 1) * expertPerRank + groupIdx]` is zero for every group. | Confirmed deviation: external nonzero counts are not visible at the exact contiguous base consumed by `GetCumsumForMMAIV`. |
| Expert totals output | Official core 0 copies `cumsumMM[(EP - 1) * expertPerRank]` to `ExpertTokenNums`, giving per-expert totals used as public debug metadata. | Debug output reports `expert_token_total` from host input, but kernel loop-stats sees `total_active_rows: 0`. | Kernel-side cumsum must be fixed before treating any GMM2 numerical output as meaningful. |
| GMM2 AIC scheduling | Official `GMM2` reads `currentM = cumsumMM((EP - 1) * expertPerRank + groupIdx)`, clips to `maxOutputSize`, doubles rows for INT4, then schedules MMAD tiles. | Loop-stats sentinel reaches the official GMM2-only entry but records `total_core_loops: 0` and `groups_with_work: 0`. | Gate B is blocked by scheduling state, not by raw C2, dequant, or checkpoint weights. |
| AIC to AIV synchronization | Official SwiGLU/hidden-quant phase signals `SYNCFLAGV2C`; isolated debug mode synthesizes the same flags through `SignalGMM2OnlyReady` before `CombineV2`. | Signal path exists and is reached only after seed/copy; it cannot schedule or dequant rows while cumsum is zero. | Do not further debug flag timing until cumsum schedules nonzero GMM2 tiles. |
| AIV dequant/readback routing | Official `CombineV2` reads the same `cumsumMM`; `BlockEpilogue2` also reads `tokenPerExpert(tokenPerExpertLayout(dstEpIdx, rank, groupIdx))` and `preSumBeforeRank`. | Debug mode zeroes `preSumBeforeRank` for the single-rank case and uses the official `BlockEpilogue2` path, including the raw-C2 sentinel branch. | Gate C remains blocked until Gate B produces nonzero raw AIC tiles. |
| Production SVDQ operator | Not applicable; official W4A8 production/debug path remains the source of truth. | `DispatchFFNCombineW4A8SVDQ` host tiling remains fail-closed. | No production SVDQ behavior change is allowed from this table alone. |

### Latest State Probe - 2026-06-26T13:45Z

Two debug-only token-state behavior attempts were built, installed, and tested after the required official-vs-debug table:

- Cumsum seed attempt: copy external counts to both the contiguous `rank * expertPerRank` base and the official `tokenPerExpertLayout(rank, 0, 0)` base, then call the official `GetCumsumForMMAIV`.
- EP=1 direct cumsum attempt: for the current single-rank probe, additionally copy external counts directly to `cumsumMM` because official cumulative counts should equal the external per-expert counts when `EP == 1`.

Both attempts compiled and installed, but neither moved GMM2 scheduling:

- Seed attempt summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_loop_stats_probe_after_cumsum_seed_fix.json`
- EP=1 cumsum summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_loop_stats_probe_after_ep1_cumsum.json`
- Both showed `total_active_rows: 0`, `total_core_loops: 0`, `groups_with_work: 0`.

Per the appendix requirement, the current code now adds a diagnostic-only scalar state probe to the loop-stats buffer. It records `EP`, `rank`, contiguous base, layout base, external counts, `tokenPerExpert` at both bases, and `cumsumMM` at the last-rank scheduling base. This does not change production SVDQ behavior and does not use public grouped matmul.

Build/install/validation evidence for the state probe:

- Build: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_state_probe_build_kernel.log`
- `ascendc_ops_config.py`: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_state_probe_ops_config.log`
- `cmake --install`: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_state_probe_cmake_install.log`
- Repo-local install: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_state_probe_install_repo_root_cwd.log`, result `SUCCESS`.
- System OPP install: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_state_probe_install_system_opp_root_cwd.log`, result `SUCCESS`.
- ABI: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_state_probe_abi.log`, result `1 passed, 16 warnings`.
- Probe log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_loop_stats_state_probe.log`
- Probe summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_loop_stats_state_probe.json`

State-probe result:

- `state_probe.valid_magic: true`
- `state_probe.ep: 1`
- `state_probe.rank: 0`
- `state_probe.cumsum_base: 0`
- `state_probe.layout_base: 0`
- `state_probe.layout_base_equals_cumsum_base: true`
- `external_expert_token_nums_first32: [16, 16, 16, 16, 16, 16, 16, 16]`
- `token_per_expert_cumsum_base_first32: [0, 0, 0, 0, 0, 0, 0, 0]`
- `token_per_expert_layout_base_first32: [0, 0, 0, 0, 0, 0, 0, 0]`
- `cumsum_mm_last_rank_first32: [0, 0, 0, 0, 0, 0, 0, 0]`
- Scheduling remains zero: `total_active_rows: 0`, `total_core_loops: 0`, `groups_with_work: 0`.

Interpretation:

- The external expert-count GM pointer is valid and nonzero inside the kernel.
- The debug-only `CopyGMToGM` seeding path does not make those counts visible in `tokenPerExpert` or `cumsumMM` at the loop-stats scheduling readback.
- Stage 2.2 Gate B and Gate C remain failed. The next correction must reconcile the isolated GMM2-only debug state lifecycle with the successful official W4A8 path's producer/synchronization ownership before any raw-C2, dequant, or SVDQ epilogue work is meaningful.

### Extended Official GMM2 Lifecycle Table - 2026-06-26T14:08Z

This is the required pre-patch lifecycle table from `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md`. It supersedes the narrower 12:39Z state table for deciding the next behavioral patch.

| State or region | Official producer | Official consumer | Official initialization point | Physical GM/workspace address and offset | Row/tile stride | Flag or event | Signal timing | Wait timing | Final drain | Current debug behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| Packed hidden `gmA2I4_I8` | AIV `BlockEpilogue1` inside `DispatchAndCombine`, after GMM1 C2V wait. | AIC `GMM2(params)` reads `gmA2I4` by expert-contiguous group offsets. | `initBuffer`: `gmA2I4_I8.SetGlobalBuffer(workspaceInfo.ptrA2Int4)`. | `workspaceInfo.ptrA2Int4 = ptrWorkspace + offset after C/C2/A1 regions`; row base uses `params.layoutD1.GetOffset(rowStart, 0)`. | Row-major packed INT4 bytes, row width `problemShape.n() / 2`. | `SYNCFLAGV2C` releases GMM2 after hidden for each sync chunk is ready. | Official AIV sets `SYNCFLAGV2C` after `BlockEpilogue1` and `SyncAll`. | Official AIC waits `SYNCFLAGV2C` at group 0 and after sync-task groups. | `blockEpilogue1.Finalize()` before final debug taps and `CombineV2`. | GMM2-only debug directly copies external hidden into `gmA2I4_I8` before official lifecycle exists; state probe shows token state still zero. Next patch will replace this with full-lifecycle hidden override at this official boundary. |
| Hidden scale `gmPerTokenScale2` | AIV `BlockEpilogue1` writes per-token hidden scale with the packed hidden rows. | AIC GMM2 dequant path via `gmPerTokenScale2`, and AIV debug/readback. | `initBuffer`: `gmPerTokenScale2.SetGlobalBuffer(workspaceInfo.ptrPerTokenScale2)`. | `workspaceInfo.ptrPerTokenScale2 = ptrWorkspace + offset after ptrPerTokenScale`; row base is `rowStartThisCore`. | Vector length `maxOutputSize`; one FP32 scale per hidden row. | Same `SYNCFLAGV2C` as packed hidden. | Scale is ready before AIV releases GMM2 for the chunk. | GMM2 consumes after `SYNCFLAGV2C`. | Debug tap copies full scale after `blockEpilogue1.Finalize()`. | GMM2-only debug copies external scale once, but the path lacks official token/cumsum lifecycle. Next patch will copy external scale at the same per-chunk boundary as packed hidden. |
| `tokenPerExpert` | AIV `moe_init_routing_quant_v2` writes local rank counts into peer memory, then cross-rank gather materializes the peer matrix. | AIV cumsum helper, AIV hidden packing, AIV `BlockEpilogue2`, and official reset. | `initBuffer`: `tokenPerExpert.SetGlobalBuffer(shmem() + peermemInfo.offsetPeerTokenPerExpert)`. | `peermemInfo.offsetPeerTokenPerExpert = shmem.SegmentSize() - 2 * MB_SIZE`; official local base is `offset + tokenPerExpertLayout(rank, 0, 0) * sizeof(int32_t)`. | `Layout3D`: `dim0 * paddedExpertNumAligned + dim1 * expertPerRank + dim2`; padded row stride `paddedExpertNumAligned`. | Cross-rank sync/gather in `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2`. | After `moe_init_routing_quant_v2` and `SyncAll`. | Cumsum and row routing read after cross-rank gather. | `ResetTokenPerExpert(params.EP * paddedExpertNumAligned)` after `CombineV2`. | Standalone debug tried to seed `tokenPerExpert` with `CopyGMToGM`; state probe reads external counts as nonzero but both token bases stay zero. This proves the standalone lifecycle is not a valid official replacement. |
| `cumsumMM` | AIV core 0 calls `GetCumsumForMMAIV(tokenPerExpert, cumsumMM, expertPerRank, rank, EP)`. | AIC `GMM1`, AIC `GMM2`, AIV hidden packing, AIV `CombineV2`, expert-token output copy. | `initBuffer`: `cumsumMM.SetGlobalBuffer(workspaceInfo.ptrcumsumMM)`. | `workspaceInfo.ptrcumsumMM = ptrWorkspace + AlignUp(M, 256) * topK * sizeof(int32_t)`. | Shape `EP * expertPerRank`, contiguous row width `expertPerRank`. | AIV releases GMM1 through cross-core flag sequence after cumsum. | Official AIV sets first GMM1 flag after copying `ExpertTokenNums`. | Official AIC `GMM1` waits initial cross-core flag before reading cumsum; GMM2 reads after GMM1. | No reset before op end; workspace is per-launch scratch. | Standalone debug wrote direct EP=1 cumsum, but state probe still read zeros. Full-lifecycle mode will keep the official producer. |
| `preSumBeforeRank` | Official cross-rank gather helper computes prefix before current rank. | AIV hidden source row selection and `BlockEpilogue2`. | `initBuffer`: `preSumBeforeRank.SetGlobalBuffer(workspaceInfo.ptrSumBeforeRank)`. | `workspaceInfo.ptrSumBeforeRank = ptrWorkspace + offset after debug/raw regions`. | Shape `EP * expertPerRank`. | Same gather lifecycle as token matrix. | Produced before AIV hidden packing. | Read before and during hidden packing and final combine. | Workspace scratch. | GMM2-only debug zeros it for EP=1. Full-lifecycle mode will use the official computed value. |
| GMM2 AIC input tile state | AIC `GMM2` builds tile shapes from `cumsumMM`, `gmA2I4`, `ptrB2`, and `ptrScale2`. | `BlockMmad` W4A8 MMAD implementation. | `GMM2(params)` after official AIC `GMM1(params)`. | A input `workspaceInfo.ptrA2Int4`, B input `params.ptrB2` via `GetTensorAddr`, scale `params.ptrScale2`, C output `workspaceInfo.ptrC2`. | L1 tile `[128, 256, 1024]`, L0 tile `[128, 256, 256]`; INT4 doubles M internally. | `SYNCFLAGV2C`. | AIV releases after each hidden chunk. | AIC waits before scheduling group 0 and sync-task group boundaries. | `blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V)` at sync-task boundaries and final async drain. | Standalone debug never schedules because cumsum is zero. Full-lifecycle override preserves official scheduling. |
| GMM2 accumulator / D2 region | AIC `BlockMmad` writes GMM2 output to `gmC2`. | AIV `BlockEpilogue2` reads `gmC2` through `CombineV2`. | `initBuffer`: `gmC2.SetGlobalBuffer(workspaceInfo.ptrC2)`. | `workspaceInfo.ptrC2 = ptrWorkspace + offset after GMM1 C region`; debug raw FP32 tap may use `workspaceInfo.ptrCGMM2` when `W4A8_DEBUG`. | Row-major C tiles, active shape from cumsum, output columns `problemShape.k()`. | `SYNCFLAGC2V` from `blockMmad.Finalize`. | AIC signals when sync task group output is finalized. | AIV `CombineV2` waits through `BlockEpilogue2` protocol. | `BlockMmad::Finalize` and `BlockEpilogue2::Finalize`. | Not reached meaningfully in standalone debug because no active GMM2 tiles are scheduled. |
| C2V handoff state | AIC `GMM2` finalizes blocks with `SYNCFLAGC2V`. | AIV `CombineV2` / `BlockEpilogue2`. | `GMM2(params)` sync task boundaries. | Cross-core flag state, not ordinary GM. | Sync-task grouping via `IsSyncTask`. | `SYNCFLAGC2V`. | After GMM2 block finalize for a sync group. | AIV waits before consuming group output. | Finalize drains pending async MMAD. | Standalone debug synthesized `SYNCFLAGV2C` and relied on official `CombineV2`; because GMM2 did no work, C2V evidence was not meaningful. |
| `BlockEpilogue2` input state | AIC GMM2 writes `gmC2`; AIV passes token matrix, layoutD2, shmem, offsetD. | `BlockEpilogue2` dequantizes and routes final output. | Constructed in `DispatchAndCombine` before hidden epilogue loop. | Token matrix base `shmem() + offsetPeerTokenPerExpert`; output route peer D offset `peermemInfo.offsetD`. | Output row layout `layoutD2(m * topK, problemShape.k())`; tile N `L1TileShape::N`. | C2V wait inside official epilogue/Combine path. | After GMM2 finalizes. | During `CombineV2`. | `blockEpilogue.Finalize()`. | Standalone debug constructs the same epilogue but without valid upstream C2/GMM2 data. |
| FP32 post-dequant debug tap | `BlockEpilogue2` W4A8_DEBUG helper writes FP32 GMM2/debug data. | Python probe reads `gmm2PostDequant`. | `WorkspaceInfo`: `ptrCGMM2 = params.ptrDebugGMM2` when debug pointer is provided. | External output/debug tensor passed as `ptrDebugGMM2`. | Shape `maxOutputSize x problemShape.k()`. | Follows official C2V/dequant lifecycle. | During `CombineV2`. | After AIC GMM2 output is ready. | `BlockEpilogue2::Finalize`. | Current final tap is all zero because upstream GMM2 schedules zero tiles. Full-lifecycle override should make this a valid Gate C signal once Gate B is nonzero. |

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

## Current Handoff State - 2026-06-26T14:09:56Z

Status: IN PROGRESS. Stage 2.2 remains open, but the failure boundary changed.

This section supersedes the earlier raw-C2 and standalone GMM2-only handoff notes. The new binding constraint from `svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md` was read and applied: preserve the official `dispatch_ffn_combine_w4_a8` GMM1/GMM2/AIV lifecycle and replace only the hidden packed input and hidden scale at the official post-`BlockEpilogue1`, pre-`SYNCFLAGV2C` hidden boundary.

Source change made in this handoff:

- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h`
  - `InitGMM2OnlyFromPacked` now sets `gmm2OnlyFromPacked_ = false` for the debug op, so `svdq_w4a8_gmm2_debug_readback` enters the full official lifecycle instead of the standalone GMM2-only path.
- `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - Added `HasExternalGMM2HiddenOverride`.
  - In `DispatchAndCombine`, after official `BlockEpilogue1` produces hidden rows and before `SyncAll` plus `SYNCFLAGV2C`, core 0 copies external Stage 2.1 validated packed hidden bytes into `gmA2I4_I8` and external hidden scales into `gmPerTokenScale2`.
  - This preserves official routing, token matrix, cumsum, GMM1 scheduling, V2C release, GMM2 AIC scheduling, C2V handoff, and AIV dequant/readback.

Validation performed after rebuilding and installing the custom op:

- Build:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 cmake --build csrc/build --target svdqw4_a8_gmm2_debug_readback_ascend910b -- -B -j1`
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_official_lifecycle_override_build_kernel.log`
- Ops metadata:
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_official_lifecycle_override_ops_config.log`
- Staged install:
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_official_lifecycle_override_cmake_install.log`
- Repo-local custom OPP install:
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_official_lifecycle_override_install_repo_root_cwd.log`
- System OPP install:
  - Result: pass.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_official_lifecycle_override_install_system_opp_root_cwd.log`
- ABI/schema check:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/ut/ops/test_svdq_moe_abi.py::test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered`
  - Result: pass, `1 passed, 16 warnings`.
  - Evidence: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_official_lifecycle_override_abi.log`
- Real-device GMM2 probe:
  - Command: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASCEND_CUSTOM_OPP_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer LD_LIBRARY_PATH=/root/workspace/lza/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} python tools/svdq_w4a8_gmm2_from_mixed_hidden_probe.py --require-npu --summary-name phase_stage2_gmm2_official_lifecycle_override_probe.json`
  - Result: fail, but no longer all-zero.
  - Evidence:
    - Log: `/root/workspace/lza/svdq_clean_evidence/stage2/20260626T_stage2_gmm2_official_lifecycle_override_probe.log`
    - Summary: `/root/workspace/lza/svdq_clean_evidence/phase_stage2_gmm2_official_lifecycle_override_probe.json`

Key numerical result from the official-lifecycle override probe:

- `passed: false`
- `official_gmm2_kernel_launched: true`
- `hidden_q_packed_exact_match: true`
- `hidden_q_packed_mismatch_count: 0`
- `hidden_scale_finite: true`
- `hidden_scale_nonzero: true`
- `canonical_hidden_finite: true`
- `canonical_hidden_nonzero: true`
- `gmm2.post_dequant_active.nonzero: true`
- `gmm2.post_dequant_active.finite: true`
- `gmm2.post_dequant_active.max_abs: 0.6201786994934082`
- `gmm2.post_dequant_active.mean_abs: 0.0555962398648262`
- `gmm2.unfused_reference.passed: false`
- Reference comparison over 64 rows:
  - `max_abs: 0.725216269493103`
  - `mean_abs: 0.08050119131803513`
  - tolerance remains `max_abs <= 0.0002`, `mean_abs <= 0.00002`

Interpretation:

- The standalone GMM2-only zero-scheduling issue is bypassed by preserving the full official lifecycle. This is the first Stage 2.2 probe in this sequence with finite nonzero official GMM2 post-dequant output from the Stage 2.1 packed hidden boundary.
- Stage 2.2 Gate C still fails because the official output does not match the unfused reference.
- The packed hidden boundary is still exact and should not be reopened as a public grouped-matmul, scale guessing, or repacking task.
- The next debugging target is the remaining numerical mismatch under the official lifecycle: row/order mapping, active-row comparison window, expert group count interpretation, or the unfused reference's reconstruction of the official GMM2 dequant contract.
- Production `DispatchFFNCombineW4A8SVDQ` remains fail-closed. Do not add SVDQ BF16 projections, mixed AIV epilogues, SwiGLU integration, or hidden quantization to the production op until this official-lifecycle GMM2 numerical gate passes.

Git/worktree state for this handoff:

- Repo: `/root/workspace/lza/vllm-ascend`
- Branch: `codex/svdq-lowrank-l0-reuse-debug`
- Origin: `git@github.com:Zao-0/vllm-ascend-MoE-svdq.git`
- Intended committed files for this handoff:
  - `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h`
  - `csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp`
  - `docs/svdq_qwen35_moe_clean_implementation_stage2_report.md`
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
