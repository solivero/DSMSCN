import tensorflow as tf
from seg_model.MyModel.SiameseInception_Keras import SiameseInception
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from net_util import weight_binary_cross_entropy
from acc_util import Recall, Precision, F1_score
from dataset import dataset_train, dataset_val
input_shape = [256, 256, 3]
siam_incep = SiameseInception()
model = siam_incep.get_model(input_shape)
EPOCHS = 10
checkpoint_filepath = './checkpoints'
print(dataset_train.element_spec)

model.compile(optimizer="Adam", loss=weight_binary_cross_entropy, metrics=['accuracy', Recall, Precision, F1_score])
print(model.summary())
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss')

model.fit(dataset_train.take(1),
    validation_data=dataset_val,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback]
)

#model.save('saved_model')