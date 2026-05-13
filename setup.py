from setuptools import setup, find_packages
from pathlib import Path

THIS_DIR = Path(__file__).parent


readme_path = THIS_DIR / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="dune_evolution_tools",
    version="0.1.8",      
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "numba",  
    ],
    python_requires=">=3.9",
    author="Lucas de Freitas Pereira",
    author_email="lucas.defreitas@unican.es",
    description="Fast storm-scale dune toe erosion model (Larson et al. 2004 + Larson et al. 2016) with RK4 + optional mesh avalanching (Cohn & Anderson, 2025).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/defreitasL/dune_evolution_tools",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)