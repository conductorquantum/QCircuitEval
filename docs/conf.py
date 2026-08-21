from __future__ import annotations

project = "QCircuitEval"
author = "QCircuitEval contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

master_doc = "index"
exclude_patterns = ["_build"]
autodoc_typehints = "description"
html_theme = "alabaster"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
