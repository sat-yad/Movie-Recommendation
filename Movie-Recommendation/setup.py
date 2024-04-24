from setuptools import setup

with open("README.md","r",encoding="utf-8") as fh:
  long_description=fh.read()

AUTHOR_NAME="Satyam Yadav"
SRC_REPO='src'
LIST_OF_REQUIREMENTS=['streamlit']

setup(
  name=SRC_REPO,
  version="0.0.1",
  author=AUTHOR_NAME,
  author_email="satyadav24@gmail.com",
  description="A small exmaple package for movies recommendation system",
  long_description_content_type=long_description,
  package=[SRC_REPO],
  python_requires='>=3.10',
  install_requires=LIST_OF_REQUIREMENTS,
)
