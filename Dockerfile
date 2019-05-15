FROM nvidia/cuda:10.0-devel-ubuntu16.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /maskrcnn-benchmark

RUN apt-get update
RUN apt-get install -y libgtk2.0-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install Dependencies
RUN pip install torch==1.0.0 -f https://download.pytorch.org/whl/cu100/stable
RUN pip install torchvision

ADD requirements.txt /maskrcnn-benchmark/
RUN pip install -r requirements.txt
RUN pip install pycocotools

# Build Mask RCNN
ADD . /maskrcnn-benchmark
RUN rm -rf build
RUN python setup.py build develop

WORKDIR /maskrcnn-benchmark
