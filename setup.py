from setuptools import setup, find_packages

setup(
    name="chonkychunker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "langchain"
    ],
    author="Aravind Raju",
    author_email="aravindraju98@email.com",
    description="Ball Tree-based semantic text chunker for vector databases and LangChain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aravindraju98/chonkychunker",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
