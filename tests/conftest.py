#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This module overrides the default pytest configuration to run the import test generator script before collection.

import subprocess
import pytest
import sys
from pathlib import Path


def pytest_configure(config):
    """Run import test generator before pytest collection"""
    print("Generating import tests...")

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run the generator script
    result = subprocess.run([
        sys.executable,
        str(project_root / "tests" / "import_test_generator.py"),
        "./frm",
        "frm"
    ], cwd=str(project_root))

    if result.returncode != 0:
        pytest.exit(f"Import test generation failed with code {result.returncode}")
