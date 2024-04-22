# Use a base image that's compatible with Debian 11
FROM continuumio/miniconda3



# # Environment variables
# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# # Install prerequisites
# RUN apt-get update && apt-get install -y \
#     wget \
#     bzip2 \
#     ca-certificates \
#     git \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     mercurial \
#     openssh-client \
#     procps \
#     subversion \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# ARG INSTALLER_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# ARG SHA256SUM=96a44849ff17e960eeb8877ecd9055246381c4d4f2d031263b63fa7e2e930af1
# RUN wget "${INSTALLER_URL}" -O miniconda.sh -q \
#     && echo "${SHA256SUM} miniconda.sh" > shasum \
#     && sha256sum --check --status shasum \
#     && mkdir -p /opt \
#     && bash miniconda.sh -b -p /opt/conda \
#     && rm miniconda.sh shasum \
#     && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
#     && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
#     && echo "conda activate base" >> ~/.bashrc \
#     && find /opt/conda/ -follow -type f -name '*.a' -delete \
#     && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
#     && /opt/conda/bin/conda clean -afy


    
# Choose an appropriate CUDA version compatible with your needs
ARG CUDA_VERSION=11.8

# Debian 11 specific setup for NVIDIA CUDA
RUN apt-get update && apt-get install -y gnupg2 curl && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/7fa2af80.pub | apt-key add - && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y cuda-toolkit-$CUDA_VERSION

# Set PATH for CUDA binaries and libraries - ensure this matches the installed version
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install Jupyter notebook
RUN /opt/conda/bin/conda install jupyter -y
RUN mkdir /opt/notebooks

# This is just to avoid the token all the time!
RUN opt/conda/bin/jupyter notebook --generate-config
# COPY jupyter_notebook_config.json root/.jupyter
# Use Password "1234"

#############################################################################################################

# Set up Sklearn environment
RUN /opt/conda/bin/conda create -n sklearn -y scikit-learn
RUN /opt/conda/bin/conda install -n sklearn -y -c anaconda ipykernel
RUN /opt/conda/envs/sklearn/bin/python -m ipykernel install --user --name=sklearn
RUN /opt/conda/bin/conda install -n sklearn -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n sklearn -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n sklearn -y pandas

# For stratified kfold
RUN /opt/conda/envs/sklearn/bin/python -m pip install iterative-stratification

#############################################################################################################

# Set up GBDT environment
RUN /opt/conda/bin/conda create -n gbdt -y
RUN /opt/conda/bin/conda install -n gbdt -y -c anaconda ipykernel
RUN /opt/conda/envs/gbdt/bin/python -m ipykernel install --user --name=gbdt
RUN /opt/conda/envs/gbdt/bin/python -m pip install xgboost==1.5.0
# originaly catboost==1.0.3
RUN /opt/conda/envs/gbdt/bin/python -m pip install catboost==1.2.5
RUN /opt/conda/envs/gbdt/bin/python -m pip install lightgbm==3.3.1
RUN /opt/conda/bin/conda install -n gbdt -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n gbdt -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n gbdt -y pandas

# For ModelTrees
RUN /opt/conda/envs/gbdt/bin/python -m pip install https://github.com/schufa-innovationlab/model-trees/archive/master.zip

#############################################################################################################

# Set up Pytorch environment
# RUN /opt/conda/bin/conda create -n torch -y python=3.8 pytorch cudatoolkit=11.3 -c pytorch
RUN /opt/conda/bin/conda create -n torch -y python=3.8 pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
RUN /opt/conda/bin/conda install -n torch -y -c anaconda ipykernel
RUN /opt/conda/bin/conda install -n torch -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n torch -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n torch -y scikit-learn
RUN /opt/conda/bin/conda install -n torch -y pandas
RUN /opt/conda/bin/conda install -n torch -y matplotlib
RUN /opt/conda/bin/conda install -n torch -y -c pytorch captum
RUN /opt/conda/bin/conda install -n torch -y shap
RUN /opt/conda/envs/gbdt/bin/python -m ipykernel install --user --name=torch

# For stratified kfold
RUN /opt/conda/envs/torch/bin/python -m pip install iterative-stratification

# For TabNet
RUN /opt/conda/envs/torch/bin/python -m pip install pytorch-tabnet

# For NODE
RUN /opt/conda/envs/torch/bin/python -m pip install requests
RUN /opt/conda/envs/torch/bin/python -m pip install qhoptim

# For DeepGBM
RUN /opt/conda/envs/torch/bin/python -m pip install lightgbm==3.3.1

# For TabTransformer
RUN /opt/conda/envs/torch/bin/python -m pip install einops

#############################################################################################################

# Set up Keras environment
RUN /opt/conda/bin/conda create -n tensorflow -y tensorflow-gpu=1.15.0 keras
RUN /opt/conda/bin/conda install -n tensorflow -y -c anaconda ipykernel
RUN /opt/conda/bin/conda install -n tensorflow -y -c conda-forge optuna
RUN /opt/conda/bin/conda install -n tensorflow -y -c conda-forge configargparse
RUN /opt/conda/bin/conda install -n tensorflow -y scikit-learn
RUN /opt/conda/bin/conda install -n tensorflow -y pandas

#############################################################################################################

# For STG
RUN /opt/conda/envs/torch/bin/python -m pip install stg==0.1.2

# For NAM
RUN /opt/conda/envs/torch/bin/python -m pip install https://github.com/eunsikchoi1230/nam/archive/main.zip
RUN /opt/conda/envs/torch/bin/python -m pip install tabulate

# For DANet
RUN /opt/conda/envs/torch/bin/python -m pip install yacs

#############################################################################################################

# Download code into container
RUN git clone https://github.com/eunsikchoi1230/TabSurvey.git /opt/notebooks
# Start jupyter notebook
CMD opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=3123 --no-browser --allow-root