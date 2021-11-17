# coding: utf-8
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flearn",
    version="0.0.2",
    author="wnma3mz",
    author_email="wnma3mz@gmail.com",
    description="Federated Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wnma3mz/flearn",
    packages=setuptools.find_packages(),
    install_requires=[
        "ipython",
        "tqdm",
        "flask",
        "requests",
        "psutil",
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
)
