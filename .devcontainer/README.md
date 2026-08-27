# Development Containers

QCircuitEval ships two development-container configurations:

- `QCircuitEval`: default CPU container for normal development, docs, linting, and tests.
- `QCircuitEval GPU`: optional CUDA-Q GPU container based on NVIDIA's CUDA-Q image.

Both containers run `.devcontainer/setup.sh` after creation. The setup installs
the repo's dev and docs dependencies with `uv` and installs pre-commit hooks.
