#!/usr/bin/env python3
"""Print the canonical product-CB menu with optional policy suffixes."""
from __future__ import annotations

import sys

from prismaquant.cb_layout import product_format_menu


if __name__ == "__main__":
    print(product_format_menu(*sys.argv[1:]))
