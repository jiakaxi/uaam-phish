from setuptools import setup, find_packages

setup(
    name="uaam-phish",
    version="0.1.0",
    description="UAAM-Phish: Multi-modal phishing detection system",
    author="UAAM-Phish Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2",
        "pytorch-lightning>=2.3",
        "transformers>=4.41",
        "torchmetrics>=1.0",
        "pandas>=2.1",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "tldextract>=3.4",
        "omegaconf>=2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
        ],
        "viz": [
            "matplotlib>=3.7",
            "seaborn>=0.12",
            "tqdm>=4.65",
        ],
        "all": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
            "matplotlib>=3.7",
            "seaborn>=0.12",
            "tqdm>=4.65",
        ],
    },
)
