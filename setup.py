from setuptools import setup, find_packages
setup(
    name="psifold",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "h5py>=3.0.0", "torch>=1.6", "tqdm", "pytest"],
    python_requires="~=3.6",
    entry_points = {
        "console_scripts" : ["run_rgn=psifold.scripts.run_rgn:main",
                             "run_psifold=psifold.scripts.run_psifold:main",
                             "proteinnet2hdf=psifold.scripts.proteinnet2hdf:main"]
        }
)
