To build from dockerhub using apptainer:
apptainer build ubuntu_20.04_ivp_2680_learn.sif docker://everardog/ubuntu_20.04_ivp_2680_learn:latest

To drop into a shell in the apptainer instance:
apptainer shell --pwd /tmp --no-home --cleanenv --containall --writable-tmpfs ubuntu_20.04_ivp_2680_learn.sif
