import sys
import subprocess
import re

from setuptools import find_packages, setup

# Base requirements for all platforms
install_requires = [
  "aiohttp>=3.10.11",
  "aiohttp_cors>=0.7.0",
  "aiofiles>=24.1.0",
  "grpcio>=1.71.0",
  "grpcio-tools>=1.71.0",
  "Jinja2>=3.1.4",
  "numpy>=2.0.0",
  "nuitka>=2.5.1",
  "opencv-python>=4.10.0.84",
  "pillow>=10.4.0",
  "prometheus-client>=0.20.0",
  "protobuf>=5.28.1",
  "psutil>=6.0.0",
  "pydantic>=2.9.2",
  "requests>=2.32.3",
  "rich>=13.7.1",
  "scapy>=2.6.1",
  "tqdm>=4.66.4",
  "transformers>=4.46.3",
  "uuid>=1.30",
  "accelerate>=0.34.2",
  "pytest>=8.3.3",
  "pytest-asyncio>=0.24.0",
  "scapy>=2.6.1",
  "uvloop==0.21.0",
  "mlx>=0.28.0",
  "mlx-metal>=0.28.0"
]

extras_require = {
  "formatting": ["yapf==0.40.2",],
}

# Check if running non-Apple
if not sys.platform.startswith("darwin"):
  print("XOMLX is for APPLE DEVICES ONLY")
  exit()


setup(
  name="xomlx",
  version="1.0.0",
  packages=find_packages(),
  install_requires=install_requires,
  extras_require=extras_require,
  package_data={"xomlx": ["tinychat/**/*"]},
  entry_points={"console_scripts": ["xomlx = xomlx.main:run"]},
)
