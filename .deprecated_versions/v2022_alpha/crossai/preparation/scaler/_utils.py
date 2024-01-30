import numpy as np

def apply_to_splits(
    obj,
    data, 
    *, 
    func,
) -> None:
    """Apply scaling to data.

    Args:
        obj (object): Scaler class object.
        data (tuple): Data.
        func (function): Scaler callback function.

    Returns:
        tuple: Data scaled.
    """
    data = list(data)
    __shape = None
    if len(data)==6:
        x_train, x_val, x_test = data[:3] 
        __shape = x_test.shape
    elif len(data) == 4:
        x_train, x_test = data[:2] 
        __shape = x_test.shape
    elif len(data) == 2:
        x_train = data[0] 
        __shape = x_train.shape
        

    if len(__shape) == 2:
        if x_train is not None and len(x_train) != 0:
            obj.scaler = func
            obj.scaler.fit(x_train)
            x_train = obj.scaler.transform(x_train)
        if len(data) > 2:
            if len(data) == 6:
                x_val = obj.scaler.transform(x_val)
            x_test = obj.scaler.transform(x_test)

    elif len(__shape) == 3:

        if x_train is not None and len(x_train) != 0:
            obj.scaler = func
            obj.scaler.fit(x_train.reshape(len(x_train), __shape[1] * __shape[2]))
            x_train = obj.scaler.transform(x_train.reshape(len(x_train), __shape[1] * __shape[2])).reshape(len(x_train), __shape[1], __shape[2])
        if len(data) > 2:
            if len(data) == 6:
                x_val = obj.scaler.transform(x_val.reshape(len(x_val), __shape[1] * __shape[2])).reshape(len(x_val), __shape[1], __shape[2])
            x_test = obj.scaler.transform(x_test.reshape(len(x_test), __shape[1] * __shape[2])).reshape(len(x_test), __shape[1], __shape[2])

    elif len(__shape) > 3:

        if x_train is not None and len(x_train) != 0:
            obj.scaler = {}
            for i, axis_data in enumerate(np.rollaxis(x_train, obj._scale_axis)):
                _init_shape = axis_data.shape
                _new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                axis_data = axis_data.reshape(_new_shape)
                obj.scaler[i] = func
                obj.scaler[i].fit(axis_data) 
                np.rollaxis(x_train, obj._scale_axis)[i] = obj.scaler[i].transform(axis_data).reshape(_init_shape)

        if len(data) > 2:
            for i, axis_data in enumerate(np.rollaxis(x_test, obj._scale_axis)):
                _init_shape = axis_data.shape
                _new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                axis_data = axis_data.reshape(_new_shape)
                np.rollaxis(x_test, obj._scale_axis)[i] = obj.scaler[i].transform(axis_data).reshape(_init_shape)

            if len(data)==6:
                for i, axis_data in enumerate(np.rollaxis(x_val, obj._scale_axis)):
                    _init_shape = axis_data.shape
                    _new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                    axis_data = axis_data.reshape(_new_shape)
                    np.rollaxis(x_val, obj._scale_axis)[i] = obj.scaler[i].transform(axis_data).reshape(_init_shape)

    if len(data) == 6:
        data[0] = x_train
        data[1] = x_val
        data[2] = x_test 
    elif len(data) == 4:
        data[0] = x_train
        data[1] = x_test 
    else:
        data[0] = x_train
    # print(data[0])

    return tuple(data)



def partial_fit(
    obj,
    data, 
    *, 
    func,
    
) -> None:
    """Apply scaling to data.

    Args:
        obj (object): Scaler class object.
        data (tuple): Data.
        func (function): Scaler callback function.

    Returns:
        tuple: Data scaled.
    """
    data = list(data)
    __shape = None
    if len(data)==6:
        x_train, x_val, x_test = data[:3] 
        __shape = x_test.shape
    elif len(data) == 4:
        x_train, x_test = data[:2] 
        __shape = x_test.shape
    elif len(data) == 2:
        x_train = data[0] 
        __shape = x_train.shape
        

    if len(__shape) == 2:
        if x_train is not None and len(x_train) != 0:
            obj.scaler = func
            obj.scaler.partial_fit(x_train)

    elif len(__shape) == 3:

        if x_train is not None and len(x_train) != 0:
            obj.scaler = func
            obj.scaler.partial_fit(x_train.reshape(len(x_train), __shape[1] * __shape[2]))

    elif len(__shape) > 3:

        if x_train is not None and len(x_train) != 0:
            obj.scaler = {}
            for i, axis_data in enumerate(np.rollaxis(x_train, obj._scale_axis)):
                _init_shape = axis_data.shape
                _new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                axis_data = axis_data.reshape(_new_shape)
                obj.scaler[i] = func
                obj.scaler[i].partial_fit(axis_data) 

    if len(data) == 6:
        data[0] = x_train
        data[1] = x_val
        data[2] = x_test 
    elif len(data) == 4:
        data[0] = x_train
        data[1] = x_test 
    else:
        data[0] = x_train
    # print(data[0])

    return tuple(data)

def partial_transform(
    obj,
    data, 
    *, 
    func,
    
) -> None:
    """Apply scaling to data.

    Args:
        obj (object): Scaler class object.
        data (tuple): Data.
        func (function): Scaler callback function.

    Returns:
        tuple: Data scaled.
    """
    data = list(data)
    __shape = None
    if len(data)==6:
        x_train, x_val, x_test = data[:3] 
        __shape = x_test.shape
    elif len(data) == 4:
        x_train, x_test = data[:2] 
        __shape = x_test.shape
    elif len(data) == 2:
        x_train = data[0] 
        __shape = x_train.shape
        

    if len(__shape) == 2:
        if x_train is not None and len(x_train) != 0:
            x_train = obj.scaler.transform(x_train)
        if len(data) > 2:
            if len(data) == 6:
                x_val = obj.scaler.transform(x_val)
            x_test = obj.scaler.transform(x_test)

    elif len(__shape) == 3:

        if x_train is not None and len(x_train) != 0:
            x_train = obj.scaler.transform(x_train.reshape(len(x_train), __shape[1] * __shape[2])).reshape(len(x_train), __shape[1], __shape[2])
        if len(data) > 2:
            if len(data) == 6:
                x_val = obj.scaler.transform(x_val.reshape(len(x_val), __shape[1] * __shape[2])).reshape(len(x_val), __shape[1], __shape[2])
            x_test = obj.scaler.transform(x_test.reshape(len(x_test), __shape[1] * __shape[2])).reshape(len(x_test), __shape[1], __shape[2])

    elif len(__shape) > 3:

        if x_train is not None and len(x_train) != 0:
            obj.scaler = {}
            for i, axis_data in enumerate(np.rollaxis(x_train, obj._scale_axis)):
                _init_shape = axis_data.shape
                _new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                axis_data = axis_data.reshape(_new_shape)
                np.rollaxis(x_train, obj._scale_axis)[i] = obj.scaler[i].transform(axis_data).reshape(_init_shape)

        if len(data) > 2:
            for i, axis_data in enumerate(np.rollaxis(x_test, obj._scale_axis)):
                _init_shape = axis_data.shape
                _new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                axis_data = axis_data.reshape(_new_shape)
                np.rollaxis(x_test, obj._scale_axis)[i] = obj.scaler[i].transform(axis_data).reshape(_init_shape)

            if len(data)==6:
                for i, axis_data in enumerate(np.rollaxis(x_val, obj._scale_axis)):
                    _init_shape = axis_data.shape
                    _new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                    axis_data = axis_data.reshape(_new_shape)
                    np.rollaxis(x_val, obj._scale_axis)[i] = obj.scaler[i].transform(axis_data).reshape(_init_shape)

    if len(data) == 6:
        data[0] = x_train
        data[1] = x_val
        data[2] = x_test 
    elif len(data) == 4:
        data[0] = x_train
        data[1] = x_test 
    else:
        data[0] = x_train
    # print(data[0])

    return tuple(data)