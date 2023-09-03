from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy>=1.17.3",
    "scipy>=1.3.2",
    "scikit-learn>=1.3.0",
    "joblib>=1.1.1",
    "threadpoolctl>=3.1.0",
    "scs>=3.2.2",
]

EXTRAS_REQUIRE = {
    "tests": [
        "pytest",
    ],
}

setup(
    name="biquality-learn",
    use_scm_version={"local_scheme": "no-local-version"},
    setup_requires=["setuptools_scm"],
    description=(
        "biquality-learn is a library à la scikit-learn for Biquality Learning."
    ),
    author="Pierre Nodet",
    author_email="pierre.nodet@orange.com",
    packages=find_packages(),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="biquality-learn",
    license="BSD 3-Clause",
    url="https://biquality-learn.readthedocs.io/",
    project_urls={
        "Source Code": "https://github.com/biquality-learn/biquality-learn"
    },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
