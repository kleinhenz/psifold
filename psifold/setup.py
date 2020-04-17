from setuptools import setup, find_packages
setup(
    name="psifold",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "h5py", "matplotlib", "torch"],
    python_requires="~=3.6",
)
