from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str)->List[str]:
    '''
    This function returns the list of requirements in the file_path.
    '''
    with open(file=file_path) as f:
        requirements = f.readlines()
        requirements = [requirement.replace('\n', '') for requirement in requirements]
        
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements
    
setup(
    name="mlproject",
    version='0.0.1',
    author='PhiNguyen',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)