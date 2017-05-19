import csv
import skimage
import skimage.io
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import matplotlib.pyplot as plt

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    heading = next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size = 128

def create_histograms(samples):
    steering_angles = np.array([float(sample[3]) for sample in samples])
    plt.hist(steering_angles, 50)
    plt.savefig("histogram_input_steering_angles.png")
    steering_angles_left_right = np.concatenate((steering_angles, steering_angles + 0.2, steering_angles - 0.2))
    plt.clf()
    plt.hist(steering_angles_left_right, 50)
    plt.savefig("histogram_left_right_steering_angles.png")
    steering_angles_final = np.concatenate((steering_angles, -steering_angles))
    plt.clf()
    plt.hist(steering_angles_left_right, 50)
    plt.savefig("histogram_final_steering_angles.png")

def read_image(filename: str):
    return skimage.io.imread(filename)

def flip_horizontally(image):
    return image[:, ::-1, :]

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffled_samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffled_samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_center = read_image('./data/IMG/'+batch_sample[0].split('/')[-1])
                image_left = read_image('./data/IMG/'+batch_sample[1].split('/')[-1])
                image_right = read_image('./data/IMG/'+batch_sample[2].split('/')[-1])
                correction = 0.2
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)

                # also add flipped versions of the images with flipped steering angles
                image_center_flipped = flip_horizontally(image_center)
                image_left_flipped = flip_horizontally(image_left)
                image_right_flipped = flip_horizontally(image_right)
                images.append(image_center_flipped)
                images.append(image_left_flipped)
                images.append(image_right_flipped)
                angles.append(-steering_center)
                angles.append(-steering_left)
                angles.append(-steering_right)

            # The above code creates 6*batch_size samples.
            # In the following, we will at most yield batch_size samples at once.
            X_train, y_train = sklearn.utils.shuffle(np.array(images), np.array(angles))
            num_entries = X_train.shape[0]
            for i in range(0, 6):
                sub_offset = i*batch_size
                sub_offset_end = sub_offset+batch_size
                sub_offset_end_real = min(sub_offset_end, num_entries)
                yield X_train[sub_offset:sub_offset_end_real], y_train[sub_offset:sub_offset_end_real]
                if sub_offset_end >= num_entries:
                    break

create_histograms(samples)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # source image format

# adopted the nVidia model from the paper
# "End to End Learning for Self-Driving Cars", Bojarski et al., 2016
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))#, output_shape=(ch, row, col))
model.add(Conv2D(24, 5, strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, 5, strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, 5, strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))
model.add(Flatten())
# to cope with overfitting, dropout layers are added after each fully-connected layer
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='tanh'))

plot_model(model, to_file='model.png', show_shapes=True)

steps_per_epoch = int(len(train_samples) * 6 / batch_size)
validation_steps = int(len(validation_samples) * 6 / batch_size)
print("steps per epoch: {}, validation_steps{}".format(steps_per_epoch, validation_steps))

model.compile(loss='mse', optimizer='adam')
checkpointCallback = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.4f}.h5", save_best_only=True)
model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    epochs=20, callbacks=[checkpointCallback])
