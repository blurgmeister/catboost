# AGENTS.md

## Purpose

This repository is being adapted to support global interpolation across all leaf predictions within each oblivious tree in CatBoost.

The intended behavior is:

- Keep CatBoost's standard oblivious-tree training flow.
- At inference time, replace the hard leaf jump around split thresholds with multi-linear interpolation across the leaf values of the current oblivious tree.
- Make interpolation user-configurable per input feature.
- Let the user choose a symmetric smoothing span around each learned split point.
- Support two span modes:
  - absolute distance in the raw feature domain
  - relative distance expressed as a decimal percentage
- Support two interpolation shapes:
  - linear
  - sigmoid

This project will progress in 3 phases:
- Phase 1: Download data for regression and classification example tasks. We will fit the original implementation of catboost and output PDP/ICE plots for all features fitted. We will observe the piecewise constant nature of GBM relationships between features and target.
- Phase 2: Apply interpolation only at inference time, storing the selected feature spans and interpolation types within the model objects as built. This will not change model training yet. We will apply this to both CPU and GPU backend code, and also edit the python front end to accomodate the new hyper parameters for user use. Once code is complete, it will need recompilation, re-creating a new python wheel and then a pip install of the new code for testing and use. Testing will involve re-running the models, but with interpolation on to compare PDP/ICE plots.
- Phase 3: Here we will also alter the model training procedure to include leaf node prediction interpolation during the training process, but only for the final leaf nodes of the tree. Each individual tree will be constructed as normal (without interpolation throughout the node and split choosing process), but only to the resulting final leaf nodes (affecting subsequent trees being constructed in the boosting sequence). Again this will require re-compilation, creating a new wheel and another pip install for testing.

## Scope

Must preserve CatBoost's existing behavior when interpolation is disabled.

In practice that means:

- Default configuration must remain bit-for-bit compatible with current CatBoost behavior where feasible.
- Existing models without interpolation settings should load and evaluate exactly as before.
- Non-oblivious or converted asymmetric trees should not silently opt into this feature.
- Training quality, model export, and fast-path evaluation should degrade gracefully when interpolation is enabled rather than breaking existing APIs.

## Design Intent

Treat interpolation as a model-evaluation concern first.

The expected high-level flow is:

1. Train oblivious trees normally and keep the learned split borders and leaf values.
2. For each float feature used by a tree, define a smoothing region around each split border.
3. For each data point that passes through each tree, compute its distance from the each split point for each feature used in that tree. Express this is a decimal >=0 and <=1.
4. Use multi-linear interpolation logic to return the weighted sum of all leaf predictions for that tree. For example in a depth 2 tree with features A and B. If w_a and w_b represent the calculated weighting for feature A and feature B for each data point, and leaf node prediction values are denoted p_a_L_b_L, p_a_L_b_R, p_a_R_b_L and p_a_R_b_R (L denoting the left branch from the split point and R denoting the right branch from a split). Then return = p_a_L_b_L * (1-w_a) * (1-w_b) + p_a_L_b_R * (1-w_a) * w_b + p_a_R_b_L * w_a * (1-w_b) + p_a_R_b_R * w_a * w_b.   
This will need to be extended to higher dimensions.  

For oblivious trees this should be expressible as a product of per-depth branch weights, which is why global interpolation over all leaves is feasible.

## Preferred Parameter Shape

Keep the user-facing configuration explicit and narrowly scoped.

Recommended parameter families:

- `interpolation_enabled`
- `interpolation_type`: `linear` or `sigmoid`
- `interpolation_span_mode`: `absolute` or `relative`
- `interpolation_span_per_float_feature`
- `interpolation_features`
- `interpolation_min_span`

The exact names can change to fit CatBoost conventions, but the semantics should stay stable:

- interpolation is opt-in
- settings can be controlled per float feature
- the span is symmetric around each learned split border for that feature
- relative spans are interpreted consistently for every feature that uses them

If CatBoost parameter conventions suggest a better naming scheme, follow the existing style in the options layer rather than forcing the names above.

## Likely Code Touchpoints

Start by reading these areas before editing:

- `catboost/private/libs/options/oblivious_tree_options.h`
- `catboost/private/libs/options/oblivious_tree_options.cpp`
- `catboost/libs/model/model.h`
- `catboost/libs/model/model.cpp`
- `catboost/libs/model/cpu/formula_evaluator.cpp`
- `catboost/libs/model/cpu/evaluator_impl.cpp`
- `catboost/libs/model/eval_processing.h`
- `catboost/libs/model/eval_processing.cpp`
- `catboost/libs/model/ut/formula_evaluator_ut.cpp`

Additional areas may need updates depending on how far the feature is carried:

- Python package parameter plumbing under `catboost/python-package`
- CLI option exposure under `catboost/app`
- model serialization and JSON export helpers under `catboost/libs/model/model_export`
- GPU evaluator code under `catboost/libs/model/cuda` if parity is required

## Implementation Priorities

Prefer this order:

1. Add configuration types and validation.
2. Store interpolation settings in the model or evaluation state in a backward-compatible way.
3. Implement CPU inference for float-feature oblivious trees.
4. Add focused unit tests for exact-threshold, inside-span, and outside-span behavior.
5. Expose the feature through Python and CLI surfaces.
6. Decide whether exported standalone model code should support interpolation or explicitly reject it.

Do not start with broad package plumbing before the core evaluator behavior is correct and tested.

## Interpolation Weight Logic

actual_span = ifelse(span_type=='relative', MAX(minimum_span, span * abs(split_point)), span)
split_weight_linear = (MIN(MAX(split - actual_span, feature_value), split + actual_span) - split)/(2*actual_span)
split_weight_sigmoid = 1-(EXP(5/actual_span*(feature_value - split))+1)^(-1)


## Testing Expectations

At minimum, cover:

- interpolation disabled gives current predictions exactly
- one split, one feature, linear interpolation
- one split, one feature, sigmoid interpolation
- multi-depth oblivious tree producing weighted sums over all leaves
- per-feature enablement where one feature is smoothed and another stays hard-thresholded
- absolute span semantics
- relative span semantics
- values exactly on the border
- values exactly at the smoothing-span edges
- model serialization round-trip if settings are persisted in the model

Primary test location:

- `catboost/libs/model/ut/formula_evaluator_ut.cpp`

Add higher-level tests only after the evaluator-level behavior is stable.

## Guardrails

- Preserve fast paths when interpolation is disabled.
- Avoid hidden behavior changes for categorical, text, embedding, and CTR-based paths.
- Be explicit about whether interpolation applies only to float splits.
- If a feature reuses the same float split in multiple trees, the configured span semantics should remain consistent.
- If a relative span depends on feature scale, document precisely what it is relative to.
- Do not guess on serialization format changes; keep old models readable.

## Open Questions To Resolve Early

- Are spans defined in raw feature space before quantization, in quantized-border space, or both?
- How should sigmoid steepness be derived from the configured span?
- Should interpolation be stored in the model artifact so inference is self-contained?
- Is CPU-only support acceptable for the first version, or is GPU parity required from the start?
- Should exported C++, Python, JSON, CoreML, ONNX, and PMML formats support this feature immediately?

Record the answer to each of these in code comments or follow-up docs once decided.

## Working Style For Future Agents

- Read existing CatBoost option and serialization patterns before adding new fields.
- Match the repository's naming and validation style rather than introducing a separate mini-framework.
- Keep changes incremental and reviewable.
- Prefer evaluator unit tests with tiny synthetic oblivious trees over large end-to-end experiments for first-pass validation.
- If a behavior choice is ambiguous, document the assumption in the PR or commit message and keep the implementation narrow.

## Future Work Note: Per-Feature Interpolation Type And Span Mode

The current implementation supports:

- per-feature interpolation span values
- one global interpolation type for the whole model
- per-feature interpolation span mode

If future work should extend the backend to allow choosing interpolation type per float feature as well, then this does not look like a small evaluator-only patch. The likely scope includes:

- options layer changes in `catboost/private/libs/options` so the configuration can carry per-feature interpolation type alongside per-feature spans and types
- model runtime structure changes in `catboost/libs/model/model.h` so `TFloatFeatureInterpolationConfig` stores feature-specific type and span-mode fields instead of relying on only global `Type`
- serialization format changes in `catboost/libs/model/model.cpp` and `catboost/libs/model/flatbuffers/model.fbs` so the model artifact persists these per-feature settings while keeping old models readable
- CPU evaluator changes in `catboost/libs/model/cpu/evaluator_impl.cpp` so each float split uses the interpolation type and span mode associated with its float feature
- GPU evaluator changes in `catboost/libs/model/cuda` because the current GPU path derives simple global flags such as sigmoid vs linear; that logic would need per-feature lookups instead
- Python parameter plumbing in `catboost/python-package/catboost/core.py` and related wrappers so the user-facing API can express per-feature type/mode in a validated format
- test updates across model serialization tests, evaluator unit tests, Python option round-trip tests, and CPU/GPU parity tests

One reasonable design direction would be to extend the per-feature interpolation config object to carry:

- float feature index
- span
- interpolation type
- interpolation span mode

If this is implemented, preserve backward compatibility by:

- defining how global settings interact with per-feature overrides
- keeping old model artifacts and old Python call patterns valid

## Future work: filtering data and interpolating only what's needed

We know that only data point within the span of each feature's split point in each tree need interpolation, those outside that range can be treated as normal (with binary radix and fast lookup). 
We should implement a filtering procedure to only route data points that need interpolating, rather than interpolating everything all the time. Furthermore, any trees that have no interpolated features should simply proceed through the traditional route as well.  

## Phase 2 Implementation Progress

• Phase 2 is not complete yet. The core CPU inference path is done, and GPU source changes have now been started, but these pieces are still left:

  1. GPU inference parity.
     GPU evaluator source changes have now been added under `catboost/libs/model/cuda`, including:
      - interpolation settings copied into GPU model state
      - a separate interpolation-aware CUDA evaluation kernel for raw-float input paths
      - GPU unit-test additions for absolute-span and relative-span-with-min-span cases
      Remaining GPU work:
      - compile and run the CUDA build
      - verify numerical parity against CPU interpolation on supported cases
      - decide whether/how GPU quantized-input evaluation should support interpolation beyond the current explicit rejection
  3. Export behavior decision and implementation.
     JSON/C++/Python/CoreML/ONNX/PMML export currently compiles, but interpolation-specific handling has not been added
     or explicitly rejected.
  4. CLI exposure check.
     The options layer and plain options mapping are in place, but the command-line training surface should be verified
     end to end.
  5. Wider testing.
     We’ve validated:
      - native model_ut build
      - interpolation evaluator tests
      - full TObliviousTreeModel
      - Python source syntax
        Still missing:
      - Python package tests with built extension
      - compiled/run GPU tests
      - export tests
      - maybe a dedicated old-model compatibility check

  Current status of completed plan items:

  - configuration types and validation: done
  - model persistence for inference-time settings: done
  - CPU oblivious-tree interpolation at inference: done
  - focused evaluator tests: done
  - Python parameter plumbing: source-level done
  - compile-fix pass for model_ut: done
  - GPU evaluator source implementation: started, not yet compiled/validated

  So the main remaining work is GPU build/validation plus package/export/CLI verification.
