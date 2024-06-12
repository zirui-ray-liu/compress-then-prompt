import os 
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def setup_package():
  long_description = "prompt"
  setuptools.setup(
      name='prompt',
      version='0.0.1',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT License',
      packages=setuptools.find_packages(
          exclude=['docs', 'tests', 'scripts', 'examples']),
      dependency_links=[
          'https://download.pytorch.org/whl/torch_stable.html',
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 3',
      ],
      cmdclass={"build_ext": BuildExtension},
    #   install_requires=[
    #     'datasets==1.6.2',
    #     'scikit-learn==0.24.2',
    #     'tensorboard==2.5.0',
    #     'matplotlib==3.4.2',
    #     'transformers==4.6.0',
    #     'numpy==1.21.1'
    #   ],
  )


if __name__ == '__main__':
  setup_package()