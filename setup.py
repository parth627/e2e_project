'''
This is for setting up the requirements of the project.
'''
from typing import List
from setuptools import find_packages,setup

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]
        file_obj.close()

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(name='e2eproject',
      version='0.0.1',
      author='Parth',
      author_email='parthlangalia627@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements("requirements.txt")
      )
