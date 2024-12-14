from setuptools import setup, find_namespace_packages

setup(
    name="clona",
    version="0.1.0",
    packages=find_namespace_packages(include=["*"]),
    install_requires=[
        "gphoto2",
        "nicegui",
        "numpy",
        "opencv-python",
        "rawpy",
        "exifread",
        "matplotlib",
        "scipy"
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'pytest-asyncio>=0.21.0',
            'coverage>=7.2.0'
        ]
    }
) 