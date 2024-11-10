from setuptools import setup, find_packages

setup(
    name="starccatovae",
    version="0.0.1",
    packages=find_packages(where="src"),  # Tells setuptools to find packages inside "src"
    package_dir={"": "src"},  # This tells Python that packages are inside "src"
)