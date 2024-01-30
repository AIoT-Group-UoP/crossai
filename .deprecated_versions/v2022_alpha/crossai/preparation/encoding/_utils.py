import numpy as np
def encode(
    obj,
    data,
    *,
    func
) -> None:
    """Encode labels function.

    Args:
        obj (object): Encoder class object.
        data (tuple): Input data.
        func (function): Encoder function.

    Returns:
        tuple: Data with encoded labels.
    """

    data = list(data)
    if len(data) == 6: 
        y_train, y_val, y_test = data[3:] 
    elif len(data) == 4:
        y_train, y_test = data[2:] 
    elif len(data) == 2:
        y_train = data[1] 
    
    if len(data) > 2:
        if not isinstance(y_test, np.ndarray):
            y_test = np.array([[y_test]])

        if not isinstance(y_test[0], np.ndarray):
            y_test = np.array([y_test])

    if y_train is not None and len(y_train) != 0:
        obj.encoder = func
        obj.encoder.fit(y_train)
        y_train = obj.encoder.transform(y_train)

    if len(data) > 2:
        y_test = obj.encoder.transform(y_test)

        if len(data) == 6:
            y_val = obj.encoder.transform(y_val)

    if len(data) == 6:
        data[3] = y_train
        data[4] = y_val
        data[5] = y_test 
    elif len(data) == 4:
        data[2] = y_train
        data[3] = y_test 
    elif len(data) == 2:
        data[1] = y_train

    return tuple(data)