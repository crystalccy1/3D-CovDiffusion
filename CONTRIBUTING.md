# Contributing to 3D-CovDiffusion

Thanks for helping improve the 3D-CovDiffusion research release. Small,
reproducible changes are easiest to review.

## Before opening a change

- Search existing issues and keep each issue or pull request focused on one
  problem.
- Use the bug template for failures in commands, configurations, data,
  checkpoints, metrics, or documentation.
- Do not commit datasets, checkpoints, generated runs, access tokens, private
  data, or machine-specific paths. Released artifacts belong on Hugging Face.
- Check the repository-level `LICENSE` before submitting code. If it is absent,
  discuss the proposed contribution in an issue before opening a code pull
  request.

## Set up

Follow [Getting Started](README.md#-getting-started), then verify the environment:

```bash
python reproduce.py doctor
```

## Validate a change

Run the checks relevant to the files you changed. The lightweight research-code
suite is:

```bash
python -m compileall -q \
  reproduce.py train.py evaluate.py \
  covdiffusion models utils scripts

python -m unittest discover -s tests -v
```

Changes to training, evaluation, data preparation, or checkpoints should also
document the exact command, category, seed, hardware, artifact revision, and
the smallest completed smoke test. Do not claim a full paper-result
reproduction from a smoke test.

## Pull requests

Describe what changed, why it is needed, and how it was verified. Update the
relevant documentation or manifest when behavior, configuration, artifact
hashes, or reproducibility claims change. Maintainers may ask that a large
change be split into smaller pull requests; review or merge is not guaranteed.
