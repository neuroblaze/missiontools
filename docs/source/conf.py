import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -------------------------------------------------------

project   = 'missiontools'
author    = 'Peter Kazakoff'
release   = '0.1.0'
copyright = f'2025, {author}'

# -- General configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

# Napoleon (numpy-style docstrings)
napoleon_numpy_docstring  = True
napoleon_google_docstring = False
napoleon_use_param        = False
napoleon_use_rtype        = False
napoleon_preprocess_types = True

# Autodoc
autodoc_default_options = {
    'members':          True,
    'undoc-members':    True,
    'show-inheritance': True,
    'member-order':     'bysource',
}
autodoc_typehints = 'description'

# nbsphinx — render notebooks with their existing outputs, never re-execute
nbsphinx_execute = 'never'

# Cross-project links
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable/', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy/', None),
}

# -- HTML output ---------------------------------------------------------------

html_theme      = 'furo'
html_title      = 'missiontools'
html_static_path = ['_static']

exclude_patterns = ['_build', '**.ipynb_checkpoints']
