import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ident_toolbox',
    version='0.9',
    author='Renato Naville Watanabe',
    author_email='renato.watanabe@ufabc.edu.br',
    description='Nonlinear system identification package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=setuptools.find_packages(),
)