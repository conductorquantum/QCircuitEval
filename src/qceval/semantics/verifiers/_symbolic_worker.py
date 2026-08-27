"""Compatibility entry point for the relocated symbolic source parser."""

from qceval.frameworks.qiskit.symbolic import main

if __name__ == "__main__":
    raise SystemExit(main())
