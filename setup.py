from setuptools import setup

setup(name='minimal_sara',
      version='0.1',
      description='Minimal Fast SARA code',
      url='https://github.com/ThomasLecocq/minimal_sara',
      author='Thomas Lecocq',
      author_email='tom@asktom.be',
      license='LGPL',
      packages=['minimal_sara'],
      zip_safe=False,
      entry_points={
          'console_scripts': ['msara=minimal_sara.command_line:main'],
      }
      )
