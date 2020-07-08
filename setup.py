from setuptools import setup, find_packages
setup(
    name="psifold",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "h5py", "matplotlib", "torch>=1.5", "tqdm", "tensorflow", "pytest"],
    python_requires="~=3.6",
    entry_points = {
        "console_scripts" : ["run_rgn=psifold.scripts.run_rgn:main",
                             "proteinnet2hdf=psifold.scripts.proteinnet2hdf:main"]
        }
)
