FROM tensorflow/tensorflow
WORKDIR /app
RUN pip install numpy opencv-python keras
RUN apt install -y libgl1-mesa-glx
RUN ["/bin/bash"]