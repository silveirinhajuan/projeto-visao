from setuptools import setup, find_packages

setup(
    name="visao",
    version="3.0.0",
    description="VISÃO - Agentic Cognitive Architecture with Liquid Neural Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Juan Guerra",
    author_email="juan.guerra@example.com",
    url="https://github.com/silveirinhajuan/projeto-visao",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "numba>=0.56",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
        "docs": [
            "mkdocs>=1.5",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
