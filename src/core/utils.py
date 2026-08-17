def delete_nested_key(d: dict, path: str) -> None:
    """Delete a value from a nested dict using a '/'-separated key path.

    Args:
        d:    The dict to mutate.
        path: '/'-separated key path (e.g. 'model/custom_model_config/hidden_dim').
    """
    keys = path.split('/')
    current = d
    for key in keys[:-1]:
        if key in current:
            current = current[key]
        else:
            return
    last_key = keys[-1]
    if last_key in current:
        del current[last_key]


def set_nested_key(d: dict, path: str, value) -> None:
    """Set a value in a nested dict using a '/'-separated key path.

    Intermediate dicts are created if they do not exist.

    Args:
        d:     The dict to mutate.
        path:  '/'-separated key path (e.g. 'model/custom_model_config/hidden_dim').
        value: The value to set.
    """
    keys = path.split('/')
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value
