#!/usr/bin/env python3

import tqdm
import os.path
from os import listdir
from ruamel.yaml import YAML
import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
import keras.layers as kl


ANNOTATIONS_DIR="res/audio_analyser_dataset/annotations/annotations averaged per song/dynamic (per second annotations)"
FEATURE_DIR="./res/audio_analyser_dataset/mfcc_features/"


class Song:
    def __init__(self, id, features, labels):
        """
        Constructs a Song.

        :param id: The id of the Song
        :type id: int
        """

        self.id = id
        self.features = features
        self.labels = labels


def create_labels(songids, arousal_annotations, valence_annotations):
    """
    Creates the labels in the following form:
    map songid -> map time(ms) -> tuple (arousal, valence)

    :param songids:
    :param arousal_annotations:
    :param valence_annotations:
    :return:
    """
    labels = {}

    only_arousal_cols = arousal_annotations.columns.difference(valence_annotations.columns)
    only_valence_cols = valence_annotations.columns.difference(arousal_annotations.columns)

    arousal_annotations.drop(only_arousal_cols, 1, inplace=True)
    valence_annotations.drop(only_valence_cols, 1, inplace=True)

    merged_labels = pd.merge(arousal_annotations, valence_annotations, left_on=["song_id"], right_on=["song_id"], suffixes=["_arousal", "_valence"])
    merged_labels.set_index("song_id", inplace=True)

    number_of_frames = 0

    for songid in tqdm.tqdm(songids, desc="creating labels"):
        time2labels = {}

        # iterate over times and fill time2labels
        for col in merged_labels:
            arousal = False

            time = col.replace("sample_", "")

            # define time
            if time.endswith("_arousal"):
                time = time.replace("_arousal", "")
                arousal = True
            elif time.endswith("_valence"):
                time = time.replace("_valence", "")
            else:
                if col != "song_id":
                    raise ValueError("col: \"" + col + "\"")

            value = merged_labels.at[songid, col]
            if pd.isnull(value):
                continue

            # fill time into time2labels
            if time in time2labels:
                if arousal:
                    old_value = time2labels[time][1]
                    time2labels[time] = (value, old_value)
                else:
                    old_value = time2labels[time][0]
                    time2labels[time] = (old_value, value)
            else:
                number_of_frames += 1
                if arousal:
                    time2labels[time] = (value, np.NaN)
                else:
                    time2labels[time] = (np.NaN, value)
        has_nan = False
        for k, v in time2labels.items():
            if np.isnan(v[0]) or np.isnan(v[1]):
                has_nan = True
                print("Found NaN Value in labels.")
                print("songid: {}\ntime: {}".format(songid, k))
                break
        if not has_nan:
            labels[songid] = time2labels

    return labels, number_of_frames


def create_features(songids):
    """
    Returns a map songid -> map time -> [mfccs]
    :param songids:
    :return:
    """
    yaml=YAML(typ="safe")

    mfccs = {}

    for songid in tqdm.tqdm(songids, desc="Loading yaml files"):
        time2mfccs = {}
        with open(os.path.join(FEATURE_DIR, str(songid) + ".yaml"), "r") as f:
            mfcc_list = yaml.load(f)["lowlevel"]["mfcc"]
            time = 0
            for frame_mfccs in mfcc_list:
                time2mfccs[str(time) + "ms"] = frame_mfccs
                time += 500

        mfccs[songid] = time2mfccs

    return mfccs


NUM_FEATURES = 13
NUM_LABELS = 2


def create_train_data(features, labels, number_of_frames):

    train_data = np.zeros((number_of_frames, NUM_FEATURES), dtype=np.float)
    train_labels = np.zeros((number_of_frames, NUM_LABELS), dtype=np.float)

    counter = 0
    for songid in labels.keys():
        if counter == 0:
            print("songid: {}".format(songid))
        if songid not in features:
            raise AssertionError("songid \"{}\" not found in features".format(songid))
        for time in labels[songid]:
            if time not in features[songid]:
                raise AssertionError("time \"{}\" not found in features. songid: \"{}\"".format(time, songid))

            train_data[counter] = features[songid][time]
            train_labels[counter] = labels[songid][time]
            counter += 1

    return train_data, train_labels


def test():
    train_data = np.ones((200, 13), dtype=np.float)
    train_labels = np.zeros((200, 2), dtype=np.float)

    model = create_model()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.expanduser("model/test_model.hdf5"),
            save_weights_only=False)]

    model.fit(x=train_data, y=train_labels, epochs=10, batch_size=50, shuffle=True, callbacks=callbacks)
    print(model.predict(x=np.array([[1.0]*13])))


def load_test_data(filename):
    yaml=YAML(typ="safe")
    mfcc_list = None
    with open("./res/test_data/" + filename, "r") as f:
        mfcc_list = yaml.load(f)["lowlevel"]["mfcc"]

    mfcc_list = mfcc_list[int(len(mfcc_list)/10):int(len(mfcc_list)/10*9)]

    return np.array(mfcc_list)


def load_model():
    model = keras.models.load_model("./model/model.hdf5")
    return model


def test_model(filename):
    test_data = load_test_data(filename)
    model = load_model()
    result = model.predict(test_data)

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    # print(result)
    res_avg = np.average(result, axis=0)
    print("result for {}:".format(filename))
    print("arousal: {}\nvalence: {}".format(res_avg[0], res_avg[1]))


def train():
    arousal_annotations = pd.read_csv(os.path.join(ANNOTATIONS_DIR, "arousal.csv"))
    valence_annotations = pd.read_csv(os.path.join(ANNOTATIONS_DIR, "valence.csv"))

    file_list = listdir(FEATURE_DIR)

    songids = []

    # build songids
    for filename in file_list:
        if filename.endswith(".yaml"):
            songid = int(filename.replace(".yaml", ""))
            songids.append(songid)

    # songids = [2, 3]
    # songids = songids[:int(len(songids)/20)]

    labels, number_of_frames = create_labels(songids=songids,
                                             arousal_annotations=arousal_annotations,
                                             valence_annotations=valence_annotations)

    features = create_features(songids)

    data, labels = create_train_data(features, labels, number_of_frames)

    train_data = data[:int(len(data)*0.7)]
    train_labels = labels[:int(len(labels)*0.7)]
    eval_data = data[int(len(data)*0.7):]
    eval_labels = labels[int(len(labels)*0.7):]

    model = create_model()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.expanduser("model/model.hdf5"),
            save_weights_only=False)]

    model.fit(x=train_data,
              y=train_labels,
              validation_data=(eval_data, eval_labels),
              epochs=40,
              batch_size=50,
              shuffle=True,
              callbacks=callbacks)


def create_model():
    model = Sequential()

    model.add(kl.Dense(units=50,
                       input_dim=NUM_FEATURES,
                       kernel_regularizer=keras.regularizers.l2(0.001),
                       bias_initializer="he_normal",
                       activation="relu"))

    model.add(kl.Dropout(0.2))

    model.add(kl.Dense(units=60,
                       kernel_initializer="glorot_uniform",
                       kernel_regularizer=keras.regularizers.l2(0.001),
                       activation="relu"))

    model.add(kl.Dense(units=50,
                       kernel_initializer="glorot_uniform",
                       kernel_regularizer=keras.regularizers.l2(0.001),
                       activation="relu"))

    model.add(kl.Dense(units=2,
                       kernel_initializer="glorot_uniform",
                       kernel_regularizer=keras.regularizers.l2(0.001),
                       # activation="tanh",
                       name="prediction"))

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss="mse")

    return model


if __name__ == "__main__":
    # train()
    filenames = os.listdir("./res/test_data/")
    for filename in filenames:
        if filename.endswith(".yaml"):
            print()
            test_model(filename)
    # test()
