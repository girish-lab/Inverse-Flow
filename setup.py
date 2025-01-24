from setuptools import setup

setup(
    name='Inverse-Flow',
    version='0.0.1',
    description="Inverse-Flow",
    author="Sandeep Nagar",
    author_email='',
    packages=[
        'inf'
    ],
    entry_points={
        'console_scripts': [
            'inf=inf.cli:main',
        ]
    },
    python_requires='>=3.9',
)