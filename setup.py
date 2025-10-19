from setuptools import setup, find_packages

setup(
    name="uaam-phish",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "tldextract",
        "torch",
        "pytorch-lightning",
    ],
)
