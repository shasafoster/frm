from distutils.core import setup

# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56

setup(
  name = 'frm',         # How you named your package folder (MyLib)
  packages = ['frm'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='Mozilla Public License Version 2.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A python package for quantitative finance and derivative pricing',   # Give a short description about your library
  author = 'shasa',                   # Type in your name
  author_email = 'your.email@domain.com',      # Type in your E-Mail
  url = 'https://github.com/shasafoster/frm',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/shasafoster/frm/archive/refs/tags/v0.0.1.tar.gz',    # copy-paste the tar.gz link here from your release
  keywords = ['finance', 'derivative', 'risk'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
    'holidays',
    'pandas',
    'numpy',
    'pandas_market_calendars',
    'datetime',
    'scipy',
    'matplotlib',
    'numba',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Mozilla Public License Version 2.0',   # Again, pick a license
    'Programming Language :: Python :: 3.10', # Specify which pyhton versions that you want to support
  ],
)