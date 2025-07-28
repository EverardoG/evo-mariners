This image is built from the image painetyler/ubuntu_20.04_ivp_2680

Pull with:
docker pull painetyler/ubuntu_20.04_ivp_2680

Build with (Add sudo if your user is not part of the docker group):
docker build --no-cache --platform=linux/amd64 -t ubuntu_20.04_ivp_2680_learn .

Update tag with:
docker tag ubuntu_20.04_ivp_2680_learn:latest everardog/ubuntu_20.04_ivp_2680_learn:latest

Push with:
docker push everardog/ubuntu_20.04_ivp_2680_learn:latest
