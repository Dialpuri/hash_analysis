import import_data as data
from dataclasses import dataclass
import os
import gemmi
import numpy as np
import parameters
import tensorflow as tf
import unet
from keras.metrics import categorical_accuracy, binary_accuracy
import matplotlib.pyplot as plt

from focal_loss import BinaryFocalLoss
import tensorflow_addons as tfa


def get_map_list(filter_: str) -> list[str]:
    return [path.path for path in os.scandir(params.maps_dir) if filter_ in path.name]


def _generate_sample(filter_: str):
    map_list = get_map_list(filter_)

    for map_path in map_list:

        pdb_code = map_path.split("/")[-1].split(".")[0].strip(filter_).strip("_")

        structure = data.import_pdb(pdb_code)
        neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()
        map_ = gemmi.read_ccp4_map(map_path).grid

        a = map_.unit_cell.a
        b = map_.unit_cell.b
        c = map_.unit_cell.c

        overlap = 8

        na = (a // overlap) + 1
        nb = (b // overlap) + 1
        nc = (c // overlap) + 1

        translation_list = []

        for x in range(int(na)):
            for y in range(int(nb)):
                for z in range(int(nc)):
                    translation_list.append([x * overlap, y * overlap, z * overlap])

        for translation in translation_list:
            sub_array = np.array(map_.get_subarray(start=translation, shape=[32, 32, 32]))
            output_grid = np.zeros((32, 32, 32))

            for i, x in enumerate(sub_array):
                for j, y in enumerate(x):
                    for k, z in enumerate(y):
                        position = gemmi.Position(translation[0] + i, translation[0] + j, translation[0] + k)
                        atoms = neigbour_search.find_atoms(position, "\0", radius=5)

                        mask = 1 if len(atoms) > 1 else 0

                        output_grid[i][j][k] = mask

            mask = output_grid.reshape((32, 32, 32, 1))
            original = sub_array.reshape((32, 32, 32, 1))

            if (mask == 1).sum() > 15_000:
                yield original, mask


def train():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    _train_gen = _generate_sample("train")
    _test_gen = _generate_sample("test")

    batch_size = 4

    spec = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_generator(lambda: _train_gen, output_signature=(spec, spec)).batch(
        batch_size=batch_size)
    test_dataset = tf.data.Dataset.from_generator(lambda: _test_gen, output_signature=(spec, spec)).batch(
        batch_size=batch_size)

    model = unet.model()
    model.summary()
    #
    # loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2,
    # from_logits=False)

    # loss = BinaryFocalLoss(gamma=2)
    loss = tf.keras.losses.BinaryCrossentropy()


    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0005)


    model.compile(optimizer=optimiser, loss=loss,
                  metrics=[binary_accuracy])

    reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1,
                                                          mode='auto',
                                                          cooldown=5, min_lr=0.0001)

    weight_path = "3dunet_weights.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min', save_weights_only=True)

    # callbacks_list = [checkpoint, reduceLROnPlat]
    callbacks_list = []
    model.fit(
        train_dataset,
        epochs=20,
        steps_per_epoch=10,
        validation_data=test_dataset,
        validation_steps=100,
        callbacks=callbacks_list,
        verbose=2
    )

    test_data = next(_test_gen)
    original = test_data[0]
    mask = test_data[1]
    prediction = model.predict(original.reshape(1,32,32,32,1))

    prediction = prediction.squeeze()

    x = np.indices(prediction.shape)[0]
    y = np.indices(prediction.shape)[1]
    z = np.indices(prediction.shape)[2]
    col = prediction.flatten()

    fig = plt.figure()
    ax3D = fig.add_subplot(1,2,1,projection='3d')
    ax3D2 = fig.add_subplot(1,2,2,projection='3d')

    p3d = ax3D.scatter(x, y, z, s=col)
    p3d = ax3D2.scatter(x, y, z, s=mask)

    plt.show()

    model.save("model_tanh")


def evaluate_training_set():
    _train_gen = _generate_sample("train")
    _test_gen = _generate_sample("test")

    count = {"0": 0, "1": 0}
    sum = {"0": 0, "1": 0}

    for train_data in _train_gen:
        density = train_data[0].squeeze()
        mask = train_data[1].squeeze()

        x = np.indices(mask.shape)[0]
        y = np.indices(mask.shape)[1]
        z = np.indices(mask.shape)[2]
        col = mask.flatten()

        fig = plt.figure()
        ax3D = fig.add_subplot(projection='3d')
        p3d = ax3D.scatter(x, y, z, s=col)

        plt.show()

        break

        # unique, counts = np.unique(mask, return_counts=True)
        #
        # occurrence_dict = dict(zip(unique, counts))
        #
        # sum["0"] += occurrence_dict[0.0]
        # sum["1"] += occurrence_dict[1.0]
        # count["0"] += 1
        # count["1"] += 1
        #
        # avg_zero = sum["0"] / count["0"]
        # avg_one = sum["1"] / count["1"]
        #
        # print("Averages so far - 0:", avg_zero, " 1:", avg_one, " Ratio = ", avg_zero / avg_one, ":1")


if __name__ == "__main__":
    params = parameters.Parameters()
    # evaluate_training_set()
    train()
