from setuptools import setup, find_packages
import re

# Function to extract version from __init__.py
def get_version():
    with open("seawater_analyzer/__init__.py", "r") as f:
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Version not found")

# Function to read requirements
def get_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()

setup(
    name="seawater-analyzer",
    version=get_version(),
    author="Danan Nave / NJ-Labs",
    author_email="nave0712@gmail.com",
    description="Backend for Seawater Composition Analysis and PHREEQC Simulation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/seawater-analyzer", # Replace with your repo URL
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha", # Or Beta, Production/Stable
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    python_requires='>=3.8', # Specify minimum Python version
    # Entry points for command-line tools (optional)
    # entry_points={
    #     'console_scripts': [
    #         'sw_analyze=seawater_analyzer.cli:main',
    #     ],
    # },
    include_package_data=True, # To include non-code files listed in MANIFEST.in
)