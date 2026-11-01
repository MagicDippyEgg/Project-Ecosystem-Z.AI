"""Packaging metadata for Project Ecosystem."""

from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
long_desc = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setup(
    name="project-ecosystem",
    version="1.0.0",
    description="A native Linux desktop artificial life simulation.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Project Ecosystem",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*", "scripts")),
    python_requires=">=3.10",
    install_requires=[
        "PySide6>=6.5",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "project-ecosystem=ecosystem.app:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: X11 Applications :: Qt",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment :: Simulation",
    ],
)
