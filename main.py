import pandas as pd
import matplotlib.pyplot as plt
import keras.preprocessing
import numpy as np


def parse_input_data(data_filename):
    input_data = pd.read_csv(data_filename, header=0, skiprows=[1])
    input_data = input_data.iloc[:, ::-1] # reverse data order

    # KVNY_daily_data = KVNY_data[KVNY_data['Time'].str.contains("07:51AM")]
    # KVNY_daily_data = KVNY_data[~KVNY_data['air_temp_high_24_hour'].isna()]

    # null_pct = KVNY_daily_data.apply(pd.isnull).sum()/KVNY_daily_data.shape[0]
    # print(null_pct)

    # Save only columns which has less than 5% omission percentage
    # valid_columns_daily_KVNY = KVNY_daily_data.columns[null_pct < .05]

    selected_columns = ['altimeter', 'air_temp', 'relative_humidity', 'wind_speed', 'wind_direction',
                        'visibility', 'sea_level_pressure']

    # Take only valid columns and make a copy
    weather_data = input_data[selected_columns].copy()
    print(weather_data)
    # weather_data.set_index('date_time', inplace=True)
    weather_data.index = input_data['date_time']
    # weather_data.index = pd.to_datetime(input_data['date_time'])
    print(input_data['date_time'])
    print(weather_data)
    print(weather_data.index)
    values = {'wind_speed': 0, 'wind_direction': 0}
    weather_data.fillna(value=values, inplace=True)
    weather_data = weather_data.interpolate()
    weather_data = weather_data.round({'air_temp': 2, 'sea_level_pressure': 1})

    weather_data.dropna(subset=["sea_level_pressure"], inplace=True)

    # check if any values are still NaN
    if weather_data.isnull().values.any():
        null_columns = weather_data.columns[weather_data.isna().any()].tolist()
        weather_data.to_csv("weather_data_check_FAIL.csv")
        raise Exception(f"Columns are null in weather_data: {*null_columns,} ")
    else:
        print("Successfully parsed data")
        try:
            weather_data.to_csv("weather_data_check.csv")
        except PermissionError:
            print("Cannot write weather_data to file")
        return weather_data


def plot_all_data(weather_data):

    weather_data_daily_mean = weather_data.groupby(weather_data.index.to_period('M')).mean(numeric_only=True)
    print(weather_data_daily_mean)
    for column in weather_data.columns:
        plt.figure(figsize=(20, 6))
        weather_data_daily_mean[column].plot()
        plt.title(column)

        plt.show()


def run_model(weather_data):
    print(weather_data.dtypes)
    weather_data = weather_data.copy()
    split_fraction = .75
    train_split = int(split_fraction * int(weather_data.shape[0]))
    print(train_split)

    past = 720
    future = 120
    learning_rate = 0.001
    batch_size = 256
    epochs = 10
    step = 6

    def normalize(data, train_split):
        data_mean = data[:train_split].mean(numeric_only=True, axis=0)
        data_std = data[:train_split].std(numeric_only=True, axis=0)
        return (data - data_mean) / data_std

    weather_data_normalized = normalize(weather_data, train_split)
    print(weather_data_normalized)
    train_data = weather_data_normalized.iloc[0: train_split - 1]
    print("Train data")
    print(train_data)
    val_data = weather_data_normalized.iloc[train_split:]
    print("Val data")
    print(val_data)

    start = past + future
    end = start + train_split

    x_train = train_data[weather_data.columns].values.astype(np.float)
    print("x_train")
    print(x_train)
    y_train = weather_data_normalized.iloc[start:end][['air_temp']]
    print("y_train")
    print(y_train)

    sequence_length = int(past / step)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(x_train,
                                                                      y_train,
                                                                      sequence_length=sequence_length,
                                                                      sampling_rate=step,
                                                                      batch_size=batch_size)

    x_end = len(val_data) - past - future

    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end][weather_data.columns].values.astype(np.float)
    y_val = weather_data_normalized.iloc[label_start:][['air_temp']]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=['accuracy'])
    model.summary()

    path_checkpoint = "model_checkpoint.weights.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )

    def visualize_loss(history, title):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    visualize_loss(history, "Training and Validation Loss")

    def show_plot(plot_data, delta, title):
        labels = ["History", "True Future", "Model Prediction"]
        marker = [".-", "rx", "go"]
        time_steps = list(range(-(plot_data[0].shape[0]), 0))
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, val in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel("Time-Step")
        plt.show()
        return

    for x, y in dataset_val.take(9):
        print(x[0][:, 1].numpy())
        print(y[0].numpy())
        print(model.predict(x)[0])
        show_plot(
            [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
            12,
            "Single Step Prediction",
        )

    print(weather_data_normalized.shape)
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)


def main():
    weather_data = parse_input_data("data/KVNY_1997-2023.csv")
    # plot_all_data(weather_data)
    run_model(weather_data)


if __name__ == '__main__':
    main()
