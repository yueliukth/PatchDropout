FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ARG username

SHELL [ "/bin/bash", "--login", "-c" ]
 
# install utilities
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        sudo \
        bzip2 \
        libx11-6 \
        git \
        wget \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN adduser --disabled-password \
   --gecos "Non-root user" \
   --uid $UID \
   --gid $GID \
   --home $HOME \
   $USER

# switch to that user
USER $USER
 
# install miniconda
ENV MINICONDA_VERSION py37_4.8.2
# if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3
 
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
   chmod +x ~/miniconda.sh && \
   ~/miniconda.sh -b -p $CONDA_DIR && \
   rm ~/miniconda.sh
 
# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH
 
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
 
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
 
# create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update --name base --channel defaults conda && \
    conda create --name PatchDropout && \
    conda clean --all --yes
RUN conda activate PatchDropout
RUN conda install -c conda-forge opencv
RUN conda install -c anaconda pip
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install torchsummary
RUN pip install timm==0.4.12
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge matplotlib
RUN conda install -c anaconda pandas
RUN conda install -c anaconda numpy
RUN conda install -c conda-forge pydicom
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge scikit-image
RUN pip install kaggle
RUN conda install -c conda-forge ipywidgets
RUN pip install tensorboard
RUN pip install einops==0.3.2
RUN pip install git+https://github.com/ildoonet/pytorch-randaugment
RUN pip install torch-tb-profiler
RUN pip install tensorflow-gpu==2.6

USER root
RUN apt-get update -y
RUN conda install -c anaconda pyyaml
RUN conda install -c conda-forge vim
RUN apt-get -y install vim nano
RUN pip install pyclean
RUN pip install pandas==1.2.2
RUN env > /root/env.txt #&& cron -f
RUN sudo apt install  -y rsync grsync
RUN sudo apt install -y openssh-client
RUN sudo apt-get install -y python3-mpi4py
RUN sudo apt install -y libopenmpi-dev
RUN pip install --upgrade pip
RUN pip install mpi4py
RUN conda install -c conda-forge openjpeg
RUN conda install pillow
RUN conda install -c conda-forge gdcm
RUN pip install torchtext==0.4
RUN conda install seaborn
RUN pip install torchviz
RUN apt-get install -y graphviz
RUN pip install easydict
RUN pip install wandb
RUN pip install --upgrade scikit-learn
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python==4.1.2.30