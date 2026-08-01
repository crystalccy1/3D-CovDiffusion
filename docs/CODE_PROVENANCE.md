# Code provenance

This record separates project-specific 3D-CovDiffusion release code from the
MIT-licensed upstream implementation retained in the public source tree. Data
provenance and licensing are documented separately in [`DATA.md`](DATA.md).

## MIT-licensed upstream code

### 3D Diffusion Policy and Diffusion Policy

The following files are derived in whole or substantial part from
[3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), whose
lower-level implementation in turn uses
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy):

```text
covdiffusion/common/model_util.py
covdiffusion/common/pytorch_util.py
covdiffusion/env_runner/base_runner.py
covdiffusion/model/common/dict_of_tensor_mixin.py
covdiffusion/model/common/module_attr_mixin.py
covdiffusion/model/common/normalizer.py
covdiffusion/model/diffusion/conditional_unet1d.py
covdiffusion/model/diffusion/conv1d_components.py
covdiffusion/model/diffusion/ema_model.py
covdiffusion/model/diffusion/mask_generator.py
covdiffusion/model/diffusion/positional_embedding.py
covdiffusion/model/vision/pointnet_extractor.py
covdiffusion/policy/base_policy.py
covdiffusion/policy/dp3.py
```

The complete upstream MIT notices are preserved in `LICENSES/` and summarized
in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Project-specific release implementation

The public release includes project-specific implementations for configuration,
processed training data, self-contained evaluation records, rollout metrics,
checkpoint validation, and the end-to-end reproduction workflow. The principal
entry points are:

```text
covdiffusion/configuration.py
covdiffusion/evaluation/metric_suite.py
covdiffusion/evaluation/paper_metrics.py
covdiffusion/common/trajectory_history.py
utils/dataset/canonical_evaluation_dataset.py
utils/dataset/covdiffusion_rollout_dataset.py
utils/dataset/processed_covdiffusion_dataset.py
train.py
evaluate.py
reproduce.py
scripts/validate_train_ready_dataset.py
scripts/validate_hf_checkpoint.py
scripts/verify_selected_inference.py
tests/
```

This file describes the contents of the audited release tree; it does not
replace the license text or contributor/institutional publication approval.
