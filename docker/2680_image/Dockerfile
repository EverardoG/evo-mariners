FROM ubuntu_20.04_ivp:latest
LABEL maintainer="Tyler Paine <tpaine@mit.edu>"
  
# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

ENV PATH="/home/moos/moos-ivp-2680/bin:${PATH}"
ENV IVP_BEHAVIOR_DIRS="/home/moos/moos-ivp-2680/lib:{IVP_BEHAVIOR_DIRS}"
 
RUN git clone https://github.com/mit2680/moos-ivp-2680.git
 
# Build the 2680 course software
RUN cd "$HOME/moos-ivp-2680" && ./build.sh --minrobot --release 