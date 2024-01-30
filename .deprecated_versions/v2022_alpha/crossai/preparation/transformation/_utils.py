import numpy as np

def apply_transform(
    obj, 
    data,
    *,
    func,
) -> tuple:

    __shape = None
    if len(data) == 2:
        obj.x_train = data[0] 
        __shape = obj.x_train.shape
    elif len(data) == 6: 
        obj.x_train, obj.x_val, obj.x_test = data[:3] 
        __shape = obj.x_test.shape
    else:
        obj.x_train, obj.x_test = data[:2] 
        __shape = obj.x_test.shape

    if len(__shape) == 2:

        if obj.x_train is not None:
            obj.coefs = func
            obj.coefs.fit(obj.x_train)
            __x_train = obj.coefs.transform(obj.x_train)

        if len(data) > 2:
            __x_test = obj.coefs.transform(obj.x_test)

            if len(data) == 6: 
                __x_val = obj.coefs.transform(obj.x_val)

    elif len(__shape) > 2:
        __x_train, __x_test, __x_val = [], [], []
        if obj.x_train is not None:
            obj.coefs = {}
            for i, axis_data in enumerate(np.rollaxis(obj.x_train, obj._axis)):
                axis_data = axis_data
                __new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                axis_data = axis_data.reshape(__new_shape)
                obj.coefs[i] = func
                obj.coefs[i].fit(axis_data) 
                __x_train.append(obj.coefs[i].transform(axis_data))
            __x_train = np.swapaxes(__x_train, obj._axis, 0)

        if len(data) > 2:
            for i, axis_data in enumerate(np.rollaxis(obj.x_test, obj._axis)):
                __init_shape = axis_data.shape
                __new_shape = (axis_data.shape[0], int(np.prod(axis_data.shape[1:])))
                axis_data = axis_data.reshape(__new_shape)
                __x_test.append(obj.coefs[i].transform(axis_data))
            __x_test = np.swapaxes(__x_test, obj._axis, 0)

            if len(data) == 6: 
                for i, axis_data in enumerate(np.rollaxis(obj.x_val, obj._axis)):
                    __new_shape = (axis_data.shape[0], np.prod(axis_data.shape[1:]))
                    axis_data = axis_data.reshape(__new_shape)
                    __x_val.append(obj.coefs[i].transform(axis_data))
                __x_val = np.swapaxes(__x_val, obj._axis, 0)

    if len(data) == 6: 
        return np.asanyarray(__x_train), np.asanyarray(__x_val), np.asanyarray(__x_test )
    elif len(data) == 4:
        return np.asanyarray(__x_train), np.asanyarray(__x_test)
    elif len(data) == 2:
        return np.asanyarray(__x_train)
