from setuptools import setup

setup(name='circular_collect',
      version='0.0.1',
        packages=['circular_collect', 'circular_collect.envs'],
        install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
        ]
)