import setuptools

setuptools.setup(
        name='calicost',
        version='v1.0.0',
        python_requires='>=3.8',
        packages=['calicost'],
        package_dir={'': 'src'},
        author='Cong Ma',
        author_email='congma@princeton.edu',
        description='Allele-specific CNAs and spatial cancer clone inference',
        long_description='CalicoST infers allele-specific copy number aberrations and cancer clones in spatially resolved transcriptomics data',
        url='https://github.com/raphael-group/CalicoST',
        install_requires=[
            'numpy', 
            'scipy', 
            'pandas',
            'scikit-learn',
            'scanpy',
            'anndata',
            'numba',
            'tqdm',
            'statsmodels',
            'networkx',
            'matplotlib',
            'seaborn',
            'pysam',
            'ete3',
            'ipykernel'
        ],
        include_package_data=True
)

