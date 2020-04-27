from setuptools import setup, find_packages
setup(
    name="psifold",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "h5py", "matplotlib", "torch", "tqdm"],
    python_requires="~=3.6",
    entry_points = {
        "console_scripts" : ["psifold_train=psifold.scripts.psifold_train:main",
                             "psifold_test=psifold.scripts.psifold_test:main",
                             "proteinnet2hdf=psifold.scripts.proteinnet2hdf:main"]
        }
)
