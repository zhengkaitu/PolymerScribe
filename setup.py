from distutils.core import setup
from pathlib import Path

setup(name='PolymerScribe',
      version='1.1.1',
      description='PolymerScribe',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://github.com/zhengkaitu/PolymerScribe',
      packages=['polymerscribe', 'polymerscribe.indigo', 'polymerscribe.inference', 'polymerscribe.transformer'],
      package_dir={'polymerscribe': 'polymerscribe'},
      package_data={'polymerscribe': ['vocab/*']},
      python_requires='>=3.7',
      setup_requires=['numpy'],
      install_requires=[
        "numpy",
        "torch>=1.11.0",
        "pandas",
        "matplotlib",
        "opencv-python>=4.5.5.64",
        "SmilesPE==0.0.3",
        "OpenNMT-py==2.2.0",
        "rdkit-pypi>=2021.03.2",
        "albumentations==1.1.0",
        "timm==0.4.12"
      ])
