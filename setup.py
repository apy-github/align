from setuptools import setup, find_packages

base_name = "alignment"

setup(
    name = base_name,
    version = '0.0.0',
    description="",
    url='',
    author='Adur Pastor Yabar',
    author_email='adur.pastor@astro.su.se',
    license='GPLv3',
    include_package_data = True,
    packages=find_packages(where="./"),
    zip_safe=False,
    )
