def delete_nested_key(d, path):
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
