from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    install_requires = file.read().splitlines()
# Install source as a package so it is accessible for ray workers
setup(
    name="paper_src",
    install_requires=install_requires,
    packages=find_packages(),
)
