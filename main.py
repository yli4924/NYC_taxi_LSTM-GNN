#data analisis of NYC Taxi dataset
#Feb 25 2023

import time
from preprocessing import read_data
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models import DenseLSTM
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore")
def fit(model, optimizer, criterion):
    print("{:<8} {:<25} {:<25} {:<25}".format('Epoch',
                                              'Train Loss',
                                              'Test Loss',
                                              'Time (seconds)'))
    for epoch in range(epochs):
        model.train()
        start = time.time()
        epoch_loss = []
        # for batch in train data
        for step, batch in enumerate(train_dataloader):
            # make gradient zero to avoid accumulation
            model.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            # get predictions
            out = model(inputs)
            out.to(device)
            # get loss
            loss = criterion(out, labels)
            epoch_loss.append(loss.float().detach().cpu().numpy().mean())
            # backpropagate
            loss.backward()
            optimizer.step()
        test_epoch_loss = []
        end = time.time()
        model.eval()
        # for batch in validation data
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            # get predictions
            out = model(inputs)
            # get loss
            loss = criterion(out, labels)
            test_epoch_loss.append(loss.float().detach().cpu().numpy().mean())
        print("{:<8} {:<25} {:<25} {:<25}".format(epoch+1,
                                                  np.mean(epoch_loss),
                                                  np.mean(test_epoch_loss),
                                                  end-start))


if __name__ == '__main__':
    """
    df = pd.read_json("/Users/yanli/Downloads/TGCN_Elli/elli.json")
    print(len(df["edge_mapping"]["edge_index"]))
    edge_mapping = df["edge_mapping"].copy()

    #print(df["time_periods"])
    #print("len node feature:",len(df["node_feature"]["edge_weight"]))
    print(len(df["y"]))
    print("edge_index len 0:",len(edge_mapping["edge_index"]["0"]))
    print("edge_index len 1:",len(edge_mapping["edge_index"]["1"]))
    
    #read original data
    df = read_data("nyc_sorted.csv")
    #plot_graph(df,"map.png")
    #check_map(df)
    drivers_df = df.groupby(["hack_license"])
    second_driver_df = drivers_df.get_group(list(drivers_df.groups)[1])
    print(second_driver_df.shape)
    second_driver_df.to_csv("datasets/second_driver_df.csv",index=False)
    """
    d1_df = read_data("first_driver_df.csv")
    d1_df.drop(d1_df.columns[[0,1,2,3,5]],axis=1,inplace=True)
    first = d1_df.pop('trip_time_in_secs')
    d1_df.insert (6,'trip_time_in_secs', first)
    print("shape of dataset:",d1_df.shape)
    print("dataset:",d1_df.head())
    # ensure all data is float
    values = d1_df.values
    values = values.astype('float32')
    """
    #using function from last project to re-frame input
    reframed = series_to_supervised(d1_df, 1,1)
    print("data after reframed", reframed.shape)
    print(reframed.head())
    # drop columns we don't want to predict, also dropping the "Distance to target(t-1) column values"
    reframed.drop(reframed.columns[[7,8,9,10,11,12,13]], axis=1, inplace=True)
    print(reframed.head())
    # split into train and test sets
    values = reframed.values
    """
    # total 1020 data items, using 800 for training
    n_train_data =800
    train_df = values[:n_train_data, :]
    test_df = values[n_train_data:, :]
    # split into input and outputs
    X_train, y_train = train_df[:, :-1], train_df[:, -1]
    X_test, y_test = test_df[:, :-1], test_df[:, -1]

    #print("X_train len",len(X_train[0]))
    #print("y_train len",len(y_train))
    # convert train and test data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    # use torch tensor datasets
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    # get data loaders
    batch_size = 32
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = 32
    epochs = 1000
    lr = 0.1
    window_size = 6
    # vanilla LSTM
    model = DenseLSTM(window_size, hidden_dim, lstm_layers=1, bidirectional=False, dense=False)
    model.to(device)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.MSELoss()

    # initate training
    fit(model, optimizer, criterion)

    # get predictions on validation set
    model.eval()
    preds = []
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        out = model(inputs)
        preds.append(out)


    preds = [x.float().detach().cpu().numpy() for x in preds]
    preds = np.array([y for x in preds for y in x])

    scaler = StandardScaler()
    scaler.fit(y_test.reshape(-1,1))
    scaler.fit(preds.reshape((-1,1)))
    # plot data and predictions and applying inverse scaling on the data
    #plt.plot(pd.Series(scaler.inverse_transform(y_train.reshape(-1, 1))[:, 0]), label='train values')
    plt.plot(pd.Series(scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]), label='test values')
    plt.plot(pd.Series(scaler.inverse_transform(preds.reshape(-1, 1))[:, 0]), label='test predictions')
    plt.xlabel('Date time')
    plt.ylabel('trip_time_in_secs')
    plt.title('Vanilla LSTM Forecasts')
    plt.legend()
    plt.savefig("result/lstm.png")


















