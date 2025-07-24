FROM ubuntu_20.04_ivp_2680:latest
LABEL maintainer="Everardo Gonzalez <gonzaeve@oregonstate.edu>"
  
# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Add MOOS variables to the env
ENV PATH="/home/moos/moos-ivp-learn/bin:${PATH}"
ENV IVP_BEHAVIOR_DIRS="/home/moos/moos-ivp-learn/lib:${IVP_BEHAVIOR_DIRS}"

# Set the default entry directory to the moos user's home
WORKDIR "/home/moos"

RUN git clone https://github.com/EverardoG/moos-ivp-learn.git

# Build the moos-ivp-learn codebase
RUN cd "$HOME/moos-ivp-learn" && ./build.sh --release
