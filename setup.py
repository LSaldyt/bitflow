import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bitflow", # Replace with your own username
    version="0.4.0",
    author="Lucas Saldyt",
    author_email="lucassaldyt@gmail.com",
    description="An efficient data pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LSaldyt/bitflow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
