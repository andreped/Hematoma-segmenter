
FROM ubuntu:18.04
# FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER André Pedersen <andrped94@gmail.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#ENV PYTHON_DIR /usr/bin/python3
#ENV PATH $PYTHON_DIR/bin:$PATH
#RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
#    libhdf5-dev g++ graphviz libglib2.0-0 libxext6 libsm6 libxrender1 \
#    git mercurial subversion

#RUN dpkg --configure -a && \
#    apt-get install -f

#RUN mkdir -p $PYTHON_DIR
    #echo export PATH=$PYTHON_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    #wget --quiet https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    #/bin/bash ~/anaconda.sh -f -b -p $CONDA_DIR && \
    #rm ~/anaconda.sh

# installing python3
RUN apt-get update -y && \
    apt-get install python3-pip -y && \
    apt-get update && \
    pip3 -V && \
    pip3 install bs4 && \
    pip3 install requests && \
    apt-get install python3-lxml -y && \
    pip3 install Pillow && \
    apt-get install libopenjp2-7 -y && \
    apt-get install libtiff5 -y

# install git (OBS: using -y is conveniently to automatically answer yes to all the questions)
RUN apt-get update && apt-get install -y git


# Needed for OpenCV
#RUN apt-get install -y libgtk2.0-0

RUN useradd -m deepinfer && echo "deepinfer:deepinfer" | chpasswd && adduser deepinfer sudo

ENV PYTHON_DIR /usr/bin/python3
#RUN mkdir -p $PYTHON_DIR

RUN chown deepinfer $PYTHON_DIR -R

USER deepinfer

# Python

RUN pip3 install tensorflow==1.13.1


# install tensorflow properly on CONDA
#RUN pip3 install --upgrade pip
#RUN pip3 install numpy==1.16.4
#RUN pip3 install tqdm
#RUN pip3 install --upgrade pytest
#RUN pip3 install --upgrade coverage

#RUN pip3 install --upgrade pip && \
#    pip3 install tensorflow==1.13.1 && \
#    pip3 install progressbar2 && \
#    pip3 install SimpleITK && \
#    pip3 install ipdb pytest pytest-cov python-coveralls coverage==3.7.1 pytest-xdist pep8 pytest-pep8 #pydot_ng
#    #conda install -c menpo opencv=2.4.11 && \
#    #conda clean -yt

#RUN mkdir /home/deepinfer/.keras
    #echo { '"image_dim_ordering"': '"th"', '"epsilon"': 1e-07, '"floatx"': '"float32"', '"backend"': #'"theano"'}>/home/deepinfer/.keras/keras.json

RUN mkdir -p /home/deepinfer/github && \
    cd /home/deepinfer/github/ && \
    pip3 install toolz && \
    git clone https://github.com/andreped/hematoma-segmenter.git

# make data folder and give deepinfer access to it
RUN mkdir /home/deepinfer/data && \
    chown -R deepinfer /home/deepinfer && \
    chmod 777 /home/deepinfer -R

# Running test.sh will download the weights and keep them in docker
#RUN cd /home/deepinfer/github/hematoma-segmenter && \
#bash test.sh

EXPOSE 8888

WORKDIR "/home/deepinfer"
# CMD ["/bin/bash"]
ENTRYPOINT ["python","/home/deepinfer/github/hematoma-segmenter/fit.py"]
#ENTRYPOINT ["python3", "/home/andre/Slicer_stuff/hematoma-segmenter/fit.py"]
#ENTRYPOINT ["python","/home/deepinfer/github/prostate-segmenter/fit.py"]
