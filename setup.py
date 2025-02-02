from setuptools import setup, find_packages

classifiers = """\
Development Status :: 2 - Pre-Alpha
License :: OSI Approved :: GNU General Public License (GPL)
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Intended Audience :: Science/Research
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Mathematics
Programming Language :: Python
Programming Language :: Python :: 2
Operating System :: OS Independent"""

setup(
    name='classifip',
    version='0.2.1',
    author='Sebastien Destercke',
    author_email='sebastien.destercke@hds.utc.fr',
    packages=find_packages(),
    #['Classifip','Classifip.dataset', 'Classifip.evaluation', 'Classifip.models', 'Classifip.representations'],
    url='http://pypi.python.org/pypi/classifip/',
    license="GNU General Public License (GPL)",
    platforms = "any",
    description='Classification with Imprecise Probability methods.',
    long_description=open('README.rst').read(),
    classifiers = classifiers.split('\n'), install_requires=['numpy', 'cvxopt', 'scikit-learn', 'matplotlib', 'pandas',
                                                             'feather-format']
)
