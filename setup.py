from setuptools import setup

setup(
    name="rmc-clustering",
    version="0.1.0",
    description="Resonance Manifold Clustering",
    py_modules=["resonance_manifold_clustering", "rmc"],
    install_requires=[
        "numpy",
        "scikit-learn"
    ]
)
