from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description=f.read()
    
setup(
    name="ragit",
    version="0.2",
    packages=find_packages(),
    install_requires = ['sentence-transformers>=3.4.1', 
                        'pandas>=2.2.3', 'chromadb>=0.6.3', 
                        'setuptools>=75.8.0', 
                        'wheel>=0.45.1', 'twine>=6.1.0'] ,
    long_description=  description,
    long_description_content_type="text/markdown"
)