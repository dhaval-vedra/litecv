
from setuptools import setup, find_packages
import os

# README.md फ़ाइल पढ़ें
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "LiteCV - Lightweight Computer Vision Library"

setup(
    # मूल जानकारी
    name="litecv-vedra",
    version="0.1.0",
    packages=find_packages(exclude=['examples', 'docs', 'tests', 'logo']),
    
    # पैकेज फ़ाइलें
    package_data={
        'litecv': ['*.py'],
    },
    
    # निर्भरताएँ (Dependencies)
    install_requires=[
        "pygame>=2.6.1",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
    ],
    
    # अतिरिक्त विकल्प (Optional features)
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    
    # लेखक और लाइसेंस
    author="VEDRA",
    author_email="gametidhaval980@gmail.com",
    description="A lightweight, fast, and easy-to-use Python computer vision library",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dhaval-vedra/litecv",
    project_urls={
        "Documentation": "https://github.com/dhaval-vedra/litecv/docs",
        "Source": "https://github.com/dhaval-bedra/litecv",
        "Issues": "https://github.com/dhaval-vedra/litecv/issues",
    },
    
    # लाइसेंस
    license="MIT",
    
    # क्लासिफायर (Categories)
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Python वर्जन आवश्यकताएँ
    python_requires=">=3.10",
    
    # कमांड-लाइन टूल्स (अगर कोई हों)
    entry_points={
        'console_scripts': [
            # उदाहरण: 'litecv-demo=litecv._litecv:main',
        ],
    },
    
    # कीवर्ड्स (PyPI पर खोजने में मदद)
    keywords="computer-vision image-processing real-time camera filters opencv-alternative raspberry-pi",
    
    # अन्य मेटाडेटा
    zip_safe=False,  # पैकेज को zip में न डालें (ताकि images/logo आसानी से एक्सेस हो सकें)
)
