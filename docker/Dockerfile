
# creates virtual ubuntu in docker image
FROM ubuntu:18.04

# maintainer of docker file
MAINTAINER André Pedersen <andrped94@gmail.com>

# set language, format and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# installing python3
RUN apt-get update -y && \
    apt-get install python3-pip -y && \
    #apt-get install software-properties-common python-software-properties && \
    #apt-get update && \
    #apt-get install python3.6 && \
    apt-get update && \
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
RUN pip3 install tensorflow==1.14.0 && \
    pip3 install SimpleITK && \
    pip3 install progressbar2 && \
    pip3 install tqdm

# clone git
RUN mkdir -p /home/deepinfer/github && \
    mkdir /home/deepinfer/data && \
    cd /home/deepinfer/github/ && \
    git clone https://github.com/andreped/hematoma-segmenter.git

# need to download model
RUN fileid=fileid="1F5fyrIj3lGAFa6DzKUjs0lvzFz1gK4KO" && \
    filename="unet_model.h5" && \
    cd /home/deepinfer/github/hematoma-segmenter/hematomasegmenter/ && \
    mkdir nets && \
    cd /home/deepinfer/github/hematoma-segmenter/hematomasegmenter/nets && \
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null && \
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename} && \
    cd /home/deepinfer/github/

# make data folder and give deepinfer access to it
#RUN mkdir /home/deepinfer/data && \
#    chown -R deepinfer /home/deepinfer && \
#    chmod 777 /home/deepinfer

EXPOSE 8888

WORKDIR "/home/deepinfer"
# CMD ["/bin/bash"]
ENTRYPOINT ["python3","/home/deepinfer/github/hematoma-segmenter/fit.py"]
