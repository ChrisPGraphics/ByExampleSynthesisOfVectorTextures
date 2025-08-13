FROM python:3.11.5

COPY ./ /ByExampleSynthesisOfVectorTextures/

# cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# install python dependencies
RUN pip install -r /ByExampleSynthesisOfVectorTextures/requirements.txt
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p "/ByExampleSynthesisOfVectorTextures/sam_checkpoints/"
RUN wget -O /ByExampleSynthesisOfVectorTextures/sam_checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN find /usr -name '*.pyc' -delete

WORKDIR /ByExampleSynthesisOfVectorTextures/
