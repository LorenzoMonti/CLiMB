from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="CLiMB",
    version="0.1.4",
    author="Lorenzo Monti",
    author_email="lorenzomonti42@gmail.com",
    description="CLustering In Multiphase Boundaries (CLIMB) - A two-phase clustering algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LorenzoMonti/CLiMB",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/CLiMB/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
        "hdbscan>=0.8.27",  # Only if using HDBSCAN
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
)