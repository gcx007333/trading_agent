# C:\Users\Administrator\trading_agent\setup.py
from setuptools import setup, find_packages

setup(
    name="trading-data",
    version="0.1.0",
    packages=find_packages(where="src"),  # 指定包在src目录
    package_dir={"": "src"},  # 告诉setuptools包目录映射
    author="Dongqi Ma",
    author_email="gcx007333@yahoo.co.jp",
    description="A trading agent by XGBoost for financial market prediction.",
    long_description_content_type="text/markdown",
    url="https://github.com/madongqi/trading",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
        ],
    },
)