from setuptools import setup, find_packages

setup(
    name="stock-analyzer",
    version="1.0.0",
    description="AI-powered stock analysis system with comprehensive reporting",
    author="Hiren Sai Vellanki",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.18",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "openpyxl>=3.1.2",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "openai>=0.28.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "ta>=0.10.2",
        "colorlog>=6.7.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "stock-analyzer=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)