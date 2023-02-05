FROM python:3.11

# Pytorch
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Other large, compiled things
RUN pip3 install scikit-learn scikit-image opencv-python-headless scipy

# Other stuff
RUN pip3 install ipython matplotlib tqdm tensorboardx click \
    roboticstoolbox-python ipykernel mediapy seaborn gym plotly \
    jupyter spatialmath-python machinevision-toolbox-python ipywidgets

RUN pip3 install pygame --pre

RUN useradd -rm -d /home/worker -s /bin/bash -g root -G sudo -u 1000 worker
USER worker
