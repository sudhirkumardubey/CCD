from setuptools import setup, find_packages

setup(
    name="ccd",
    version="0.1.0",
    author="Sudhir Kumar Dubey",
    description="Centrifugal Compressor Design Framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "CoolProp>=6.4.1",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
)