# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2019  Baidu.com, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
"""
Setup script.

Authors: zhangshuai28(zhangshuai28@baidu.com)
Date:    2020/06/22 11:48:37
"""
from io import open
import pkg_resources
import setuptools
from setuptools import setup
from setuptools.command.install import install

# 判断paddle安装版本，对版本进行设置
install_requires = []
try:
    import paddle
    # 若版本太低，设置版本的更新
    if paddle.__version__ < '2.3':
        installed_packages = pkg_resources.working_set
        paddle_pkgs = [i.key for i in installed_packages if "paddle" in i.key]

        if "paddlepaddle-gpu" in paddle_pkgs:
            install_requires = ['paddlepaddle-gpu>=2.3']
        elif "paddlepaddle" in paddle_pkgs:
            install_requires = ['paddlepaddle>=2.3']

except ImportError:
    install_requires = ['paddlepaddle>=2.3']
try:
    import LAC
    # 若版本太低，设置版本的更新
    if LAC.version < '2.1':
        install_requires.append('LAC>=2.1')
except ImportError:
    install_requires.append('LAC>=2.1')

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()
setup(
    name="ddparser",
    version="1.0.8",
    author="Baidu NLP",
    author_email="nlp-parser@baidu.com",
    description="A chinese dependency parser tool by Baidu NLP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baidu/ddparser",
    install_requires=install_requires,
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    include_package_data=True,
    platforms="any",
    keywords=("ddparser chinese depencency parser"),
    license='Apache 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)