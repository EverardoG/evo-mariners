FROM ubuntu:20.04
LABEL maintainer = Tyler Paine <tpaine@mit.edu>
  
# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Add MOOS variables to the env
ENV PATH="/home/moos/moos-ivp/bin:${PATH}"
ENV IVP_BEHAVIOR_DIRS="/home/moos/moos-ivp/lib"
 
# Install required MOOS dependencies
RUN apt-get update -y && \
   DEBIAN_FRONTEND=noninteractive \
   apt-get install -y cmake build-essential git && \
   apt-get clean

# Make a user to run the MOOS apps and set as default user
RUN useradd -m -p "moos" moos && usermod -aG sudo moos
USER moos
 
# Set the default entry directory to the moos user's home
WORKDIR "/home/moos"
 
RUN git clone https://github.com/moos-ivp/moos-ivp.git
 
# Build the MOOS-IvP tools
RUN cd "$HOME/moos-ivp" && ./build-moos.sh --minrobot --release && ./build-ivp.sh --nogui