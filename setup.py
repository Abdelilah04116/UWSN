### Fichier: setup.py

"""
Configuration d'installation pour le projet UWSN PPO
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="uwsn-ppo-routing",
    version="1.0.0",
    author="Votre Nom",
    author_email="votre.email@example.com",
    description="Optimisation de routage dans les réseaux de capteurs sous-marins avec PPO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-username/uwsn-ppo-routing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "uwsn-demo=demo:main",
            "uwsn-train=src.ppo_train:main",
            "uwsn-test=test_installation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "underwater",
        "sensor",
        "network",
        "routing",
        "reinforcement",
        "learning",
        "ppo",
        "acoustic",
        "optimization",
    ],
    project_urls={
        "Bug Reports": "https://github.com/votre-username/uwsn-ppo-routing/issues",
        "Source": "https://github.com/votre-username/uwsn-ppo-routing",
        "Documentation": "https://github.com/votre-username/uwsn-ppo-routing#readme",
    },
)
