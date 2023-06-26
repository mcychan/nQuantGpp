FROM ubuntu:22.04
WORKDIR /tmp
RUN apt update -y
RUN apt install -y build-essential cmake g++ libomp-dev
RUN DEBIAN_FRONTEND="noninteractive" apt install -y libopencv-dev
ADD . /tmp/nQuantGpp
WORKDIR /tmp/nQuantGpp
RUN cmake -S . -B ../build
RUN cmake --build ../build
RUN cp *.jpg /tmp/build/nQuantGpp/
WORKDIR /tmp/build/nQuantGpp
# docker system prune -a
# docker build -t nquantgpp .
# docker run -it nquantgpp bash
# docker cp <containerId>:/file/path/within/container /host/path/target
# docker cp foo.txt <containerId>:/foo.txt
