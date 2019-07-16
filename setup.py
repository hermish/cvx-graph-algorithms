import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='cvxgraphalgs',
    version='0.1.2',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'cvxpy',
        'networkx'
    ],

    author='Hermish Mehta',
    author_email='hermishdm@gmail.com',
    keywords='graph algorithms theory convex optimization',
    description='Modern convex optimization-based graph algorithms.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hermish/cvx-graph-algorithms',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
