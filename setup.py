
import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="tawergw",                     # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="Amin Akhshi",                     # Full name of the author
    description="A template-free approach for waveform extraction of gravitational wave events",
    long_description = long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["tawergw"],             # Name of the python package
    package_dir={'':'tawergw/src'},     # Directory of the source code of the package
    install_requires=[]                     # Install other dependencies if any
)