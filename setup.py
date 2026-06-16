#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HBM4 System Modeling Platform - Setup Configuration

High Bandwidth Memory (HBM) system-level simulation platform for chip design
exploration and RTL-Python verification alignment.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read long description from README
this_dir = Path(__file__).parent.resolve()
readme_file = this_dir / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements = []
req_file = this_dir / "requirements.txt"
if req_file.exists():
    with open(req_file, encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="hbm4-model",
    version="1.0.0",
    description="HBM4 System Modeling Platform for chip design exploration and verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HBM System Team",
    author_email="hbm-team@dysonai.com",
    url="https://github.com/dysonaicom-svg/hbm4-model",
    project_urls={
        "Documentation": "https://github.com/dysonaicom-svg/hbm4-model#readme",
        "Source": "https://github.com/dysonaicom-svg/hbm4-model",
        "Tracker": "https://github.com/dysonaicom-svg/hbm4-model/issues",
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache License 2.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    keywords=[
        "HBM",
        "HBM4",
        "High Bandwidth Memory",
        "DRAM",
        "memory controller",
        "system modeling",
        "transaction-level modeling",
        "chip design",
        "verification",
        "DFI",
    ],
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "tests.benchmark",
            "tests.controller",
            "tests.controller.*",
            "tests.coverage",
            "tests.dram",
            "tests.dram.*",
            "tests.hbm4",
            "tests.hbm4.*",
            "tests.integration",
            "tests.integration.*",
            "tests.interconnect",
            "tests.interconnect.*",
            "tests.regression",
            "tests.regression.*",
            "tests.simulation",
            "tests.simulation.*",
            "tests.sim",
            "tests.sim.*",
            "tests.traffic",
            "tests.traffic.*",
            "tests.verification",
            "tests.verification.*",
            "docs",
            "docs.*",
            "verification",
            "verification.*",
            "verification.uvm",
            "verification.uvm.*",
            "verification.reference_model",
            "verification.reference_model.*",
            "rtl",
            "rtl.*",
            "research",
            "research.*",
            "model.controller.tests",
            "model.dram.tests",
            "model.benchmark",
            "model.benchmark.*",
            "model.interconnect",
            "model.interconnect.*",
            "model.traffic",
            "model.traffic.*",
            "sim.interconnect",
            "sim.interconnect.*",
            "sim.trace",
            "sim.trace.*",
        ]
    ),
    package_data={
        "model": [
            "**/*.py",
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
            "**/*.md",
        ],
        "model.controller": [
            "**/*.py",
            "**/tests/*.py",
        ],
        "model.dram": [
            "**/*.py",
            "**/tests/*.py",
        ],
        "model.hbm4": [
            "**/*.py",
            "**/*.yaml",
            "**/*.yml",
        ],
        "model.phy": [
            "**/*.py",
        ],
        "model.benchmark": [
            "**/*.py",
        ],
        "model.interconnect": [
            "**/*.py",
        ],
        "model.traffic": [
            "**/*.py",
        ],
        "sim": [
            "**/*.py",
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
            "**/results/**/*",
            "**/trace/**/*",
        ],
        "sim.interconnect": [
            "**/*.py",
        ],
        "sim.trace": [
            "**/*.py",
            "**/*.tr",
            "**/*.trace",
        ],
        "examples": [
            "**/*.py",
            "**/*.yaml",
            "**/*.yml",
        ],
        "config": [
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "pylint>=2.14.0",
            "mypy>=0.960",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
            "pytest-timeout>=2.1.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
            "pytest-timeout>=2.1.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
        ],
        "uvm": [
            "systemverilog-parser>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hbm4-sim=sim.simulator:run_simulation",
            "hbm4-simulator=sim.simulator:run_simulation",
            "hbm4-unified-sim=sim.unified_simulator:run_unified_simulation",
            "hbm4-benchmark=sim.benchmark:main",
            "hbm4-unified-bench=sim.hbm4_unified_simulator:main",
            "hbm4-report=sim.report_generator:generate_html_report",
        ],
    },
    zip_safe=False,
    test_suite="pytest",
    tests_require=["pytest>=7.0.0"],
)