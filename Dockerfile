FROM tensorflow/tensorflow
WORKDIR /app
COPY . .
RUN pip install numpy opencv-python keras
RUN apt install -y libgl1-mesa-glx
CMD ["python3", "supervised/train_Unet_model.py"]