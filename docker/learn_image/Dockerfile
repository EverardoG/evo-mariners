FROM painetyler/ubuntu_20.04_ivp_2680:latest
LABEL maintainer="Everardo Gonzalez <gonzaeve@oregonstate.edu>"
  
# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Set the default entry directory to the moos user's home
WORKDIR "/home/moos"

RUN git clone https://github.com/EverardoG/moos-ivp-learn.git

# Build the moos-ivp-learn codebase
RUN cd "$HOME/moos-ivp-learn" && ./build.sh --release
