import os
import setuptools

current_dir = os.path.dirname(os.path.abspath("__file__"))


# Get the long description from the README file
with open(os.path.join(current_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# What packages are required for this module to be executed?
try:
    with open(os.path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
        required = f.read().split("\n")
except FileNotFoundError:
    required = []

setuptools.setup(
    name="stabilizer",  # Replace with your own username
    version="1.0.2",
    author="flowerpot-ai",
    author_email="vignesh.sbaskaran@gmail.com",
    description="Stabilize and achieve excellent performance with transformers",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/flowerpot-ai/stabilizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=required,
    include_package_data=True,
)
