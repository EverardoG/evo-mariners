To build the apptainer locally:
./build_image.sh 

To build apptainer on hpc:
sbatch sbatch_build_apptainer.sh

Run command in apptainer:
apptainer exec \
  --cleanenv --containall --contain --net --network=none --writable-tmpfs \
  --bind $HOME/hpc-share:/home/moos/hpc-share \
  ubuntu_20.04_ivp_2680_learn.sif \
  touch /home/moos/hpc-share/hello-world
