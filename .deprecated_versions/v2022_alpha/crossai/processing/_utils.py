import numpy as np


def concat_split(data):

    data = list(data)
    
    if len(data) == 6:
        train_pos = [0, 3]
        val_pos = [1, 4]
        test_pos = [2, 5]
    elif len(data) == 4:
        train_pos = [0, 2]
        test_pos = [1, 3]
    
    concated = []

    for train_data, train_label in zip(data[train_pos[0]], data[train_pos[1]]):

        concated.append({
            'X': train_data.astype(float),
            'Y': train_label,
            'split': 'train'
        })

    if len(data) == 6:
        for val_data, val_label in zip(data[val_pos[0]], data[val_pos[1]]):

            concated.append({
                'X': val_data.astype(float),
                'Y': val_label,
                'split': 'val'
            })

    for test_data, test_label in zip(data[test_pos[0]], data[test_pos[1]]):

        concated.append({
            'X': test_data.astype(float),
            'Y': test_label,
            'split': 'test'
        })
    
    return concated


def split_concated(data):

    x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []

    for i in data:
        if i['split'] == 'train':
            x_train.append(i['X'][0])
            y_train.append(i['Y'])

        elif i['split'] == 'val':
            x_val.append(i['X'][0])
            y_val.append(i['Y'])

        elif i['split'] == 'test':
            x_test.append(i['X'][0])
            y_test.append(i['Y'])

    if len(x_val) == 0:
        return (np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32), np.array(y_train), np.array(y_test))
    else:
        return (np.array(x_train, dtype=np.float32), np.array(x_val, dtype=np.float32), np.array(x_test, dtype=np.float32), np.array(y_train), np.array(y_val), np.array(y_test))

