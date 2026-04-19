# RL-Acc-FEM-with-co-rotational-formulation
The data and code of my CMAME paper, which include a FEM beam code and 
"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:
#  @brief:     fem environment
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2026.04.03
###########################################################################################
"""

Requirements：
	python>=3.6
	CUDA>=10.1

Install other requirements by：
	pip install -r requirements.txt


Install pycuda:
    conda install -c conda-forge pycuda
if solve environment failed, then



for windows:
From：
	https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl
	https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda

Download the .whl files of PyOpenGL64 and pycuda like：
	PyOpenGL_accelerate‑3.1.6‑cp39‑cp39‑win_amd64.whl
	PyOpenGL‑3.1.6‑cp39‑cp39‑win_amd64.whl
	pycuda‑2021.1+cuda114‑cp39‑cp39‑win_amd64.whl
in which cp39 is the python version and cuda114 is the cuda version

Install PyOpenGL and pycuda：
	pip install PyOpenGL_accelerate-3.1.6-cp39-cp39-win_amd64.whl
	pip install PyOpenGL-3.1.6-cp39-cp39-win_amd64.whl
	pip install pycuda-2021.1+cuda114-cp37-cp37m-win_amd64.whl



for ubuntu:
    sudo apt-get install python3-opengl
    pip install PyOpenGL
