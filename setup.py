from setuptools import setup, find_packages

setup(
    name="tsf-exploration",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aeon",
        "seaborn>=0.11.0",
    ],
    author="Adarsh",
    author_email="dubeyadarshmain@gmail.com",  # Add your email if you want
    description="A time series forecasting exploration project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/inclinedadarsh/tsf-exploration",  # Add your repository URL if you have one
    python_requires=">=3.10",
)
