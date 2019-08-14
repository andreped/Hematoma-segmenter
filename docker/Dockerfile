
# creates virtual ubuntu in docker image
FROM ubuntu:18.04

# maintainer of docker file
MAINTAINER André Pedersen <andrped94@gmail.com>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

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

# give deepinfer sudo access and give deepinfer access to python directories
RUN useradd -m deepinfer && echo "deepinfer:deepinfer" | chpasswd && adduser deepinfer sudo
ENV PYTHON_DIR /usr/bin/python3
RUN chown deepinfer $PYTHON_DIR -R
USER deepinfer

# Python
RUN pip3 install tensorflow==1.13.1 && \
    pip3 install SimpleITK && \
    pip3 install progressbar2 && \
    pip3 install tqdm

# clone git
RUN mkdir -p /home/deepinfer/github && \
    mkdir /home/deepinfer/data && \
    cd /home/deepinfer/github/ && \
    git clone https://github.com/andreped/hematoma-segmenter.git

# make data folder and give deepinfer access to it
#RUN mkdir /home/deepinfer/data && \
#    chown -R deepinfer /home/deepinfer && \
#    chmod 777 /home/deepinfer

EXPOSE 8888

WORKDIR "/home/deepinfer"
# CMD ["/bin/bash"]
ENTRYPOINT ["python3","/home/deepinfer/github/hematoma-segmenter/fit.py"]
