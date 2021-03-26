FROM tensorflow/tensorflow
WORKDIR /app
RUN pip install numpy opencv-python keras imageio pydensecrf sklearn tensorflow_io
RUN apt install -y libgl1-mesa-glx
RUN ["/bin/bash"]