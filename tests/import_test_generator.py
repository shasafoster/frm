#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script generates pytest code for testing imports in a Python package.


import ast
import importlib
import os
import sys
from pathlib import Path
from typing import Set, List, Dict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImportAnalyzer:
    def __init__(self, package_root: str, package_name: str):
        self.package_root = Path(package_root)
        self.package_name = package_name
        self.imports: Dict[str, Set[str]] = {}
        self.exports: Dict[str, Set[str]] = {}

    def analyze_file(self, file_path: Path) -> None:
        """Analyze imports and exports in a Python file"""
        relative_path = file_path.relative_to(self.package_root)
        module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')

        try:
            # Try UTF-8 first
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1
            with open(file_path, encoding='latin-1') as f:
                content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return

        # Collect imports
        imports = set()
        exports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                for name in node.names:
                    if name.name != '*':
                        exports.add(name.name)

        self.imports[module_path] = imports
        self.exports[module_path] = exports

        logger.info(f"Analyzed {module_path}")
        logger.info(f"Imports: {imports}")
        logger.info(f"Exports: {exports}")

    def scan_package(self) -> None:
        """Scan all Python files in the package"""
        for file_path in self.package_root.rglob('*.py'):
            if not any(part.startswith('_') for part in file_path.parts[:-1]):
                self.analyze_file(file_path)

    def generate_import_tests(self) -> str:
        """Generate pytest code for testing imports"""
        test_code = [
            "import pytest",
            "import importlib",
            "",
            "# Auto-generated import tests",
            "",
            "@pytest.mark.parametrize('module_path', [",
        ]

        # Add test for each module
        for module_path in sorted(self.imports.keys()):
            test_code.append(f"    '{self.package_name}.{module_path}',")

        test_code.extend([
            "])",
            "def test_module_imports(module_path):",
            '    """Test that each module can be imported."""',
            "    importlib.import_module(module_path)",
            "",
            "@pytest.mark.parametrize('module_path,name', [",
        ])

        # Add test for each export
        for module_path, exports in sorted(self.exports.items()):
            for export in sorted(exports):
                test_code.append(
                    f"    ('{self.package_name}.{module_path}', '{export}'),")

        test_code.extend([
            "])",
            "def test_specific_imports(module_path, name):",
            '    """Test that specific names can be imported from modules."""',
            "    module = importlib.import_module(module_path)",
            "    assert hasattr(module, name), f'{module_path} is missing export {name}'",
        ])

        return "\n".join(test_code)


import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate import tests for a Python package')
    parser.add_argument('package_root', help='Root directory of the package')
    parser.add_argument('package_name', help='Name of the package')

    # Parse arguments, but if no args provided (e.g. in PyCharm console), use defaults
    try:
        args = parser.parse_args()
        package_root = args.package_root
        package_name = args.package_name
    except SystemExit:
        # If running in PyCharm console without args, use defaults
        print("No command line args found, using defaults: ./frm and frm")
        package_root = './frm'
        package_name = 'frm'

    analyzer = ImportAnalyzer(package_root, package_name)
    analyzer.scan_package()

    output_dir = Path('tests/integration')
    output_dir.mkdir(parents=True, exist_ok=True)

    test_code = analyzer.generate_import_tests()
    output_file = output_dir / 'test_imports.py'

    with open(output_file, 'w') as f:
        f.write(test_code)

    logger.info(f"Generated import tests in {output_file}")


if __name__ == '__main__':
    main()


