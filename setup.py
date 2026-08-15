from setuptools import setup, find_packages

setup(
    name="hox",
    version="0.1.0",
    description="Onnx LLM inference engine",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "Jinja2 >= 3.1.6",
        "onnx>=1.19.1",
        "onnxruntime>=1.23.1",
        "tokenizers>=0.22.2",
        "huggingface_hub>=1.25.1",
    ],
)

