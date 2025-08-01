To build from dockerhub using apptainer:
apptainer build --force --fakeroot ubuntu_20.04_ivp_2680_learn.sif docker://everardog/ubuntu_20.04_ivp_2680_learn:latest

To drop into a shell in the apptainer instance:
apptainer shell --cleanenv --containall --contain --net --network=fakeroot --fakeroot --bind /home/ever.linux/hpc-share:/home/moos/hpc-share --writable-tmpfs /Users/ever/evo-mariners/apptainer/ubuntu_20.04_ivp_2680_learn.sif

To run a single command in apptainer
apptainer exec --cleanenv --containall --contain --net --network=fakeroot --fakeroot --bind /home/ever.linux/hpc-share:/home/moos/hpc-share --writable-tmpfs /Users/ever/evo-mariners/apptainer/ubuntu_20.04_ivp_2680_learn.sif /bin/bash -c