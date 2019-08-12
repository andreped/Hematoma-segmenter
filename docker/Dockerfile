
FROM ubuntu:18.04
# FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER André Pedersen <andrped94@gmail.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libhdf5-dev g++ graphviz libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

#RUN dpkg --configure -a && \
#    apt-get install -f

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -f -b -p $CONDA_DIR && \
    rm ~/anaconda.sh


# Needed for OpenCV
#RUN apt-get install -y libgtk2.0-0

RUN useradd -m deepinfer && echo "deepinfer:deepinfer" | chpasswd && adduser deepinfer sudo

RUN mkdir -p $CONDA_DIR

RUN chown deepinfer $CONDA_DIR -R

USER deepinfer

# Python
RUN pip install numpy==1.16.4 

RUN pip install --upgrade pip && \	
    #pip install theano && \
    pip install --ignore-installed tensorflow==1.13.1 && \
    pip install progressbar2 && \
    pip install SimpleITK && \
    pip install ipdb pytest pytest-cov python-coveralls coverage==3.7.1 pytest-xdist pep8 pytest-pep8 pydot_ng && \
    #pip install git+git://github.com/fchollet/keras.git && \
    conda install -c menpo opencv=2.4.11 && \
    conda clean -yt

RUN mkdir /home/deepinfer/.keras
    #echo { '"image_dim_ordering"': '"th"', '"epsilon"': 1e-07, '"floatx"': '"float32"', '"backend"': #'"theano"'}>/home/deepinfer/.keras/keras.json

RUN mkdir -p /home/deepinfer/github && \
    mkdir /home/deepinfer/data && \
    cd /home/deepinfer/github/ && \
    pip install keras-tqdm && \
    pip install matplotlib && \
    git clone https://github.com/andreped/hematoma-segmenter.git    

# Running test.sh will download the weights and keep them in docker
#RUN cd /home/deepinfer/github/hematoma-segmenter && \
#bash test.sh

EXPOSE 8888

WORKDIR "/home/deepinfer"
# CMD ["/bin/bash"]
ENTRYPOINT ["python","/home/deepinfer/github/hematoma-segmenter/fit.py"]
#ENTRYPOINT ["python3", "/home/andre/Slicer_stuff/hematoma-segmenter/fit.py"]
#ENTRYPOINT ["python","/home/deepinfer/github/prostate-segmenter/fit.py"]
