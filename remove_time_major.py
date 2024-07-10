import h5py
import json

def remove_time_major(config):
    if isinstance(config, dict):
        config.pop('time_major', None)  
        for key, value in config.items():
            remove_time_major(value)
    elif isinstance(config, list):
        for item in config:
            remove_time_major(item)


file_path = 'keras_model.h5'

with h5py.File(file_path, 'r+') as f:

    model_config = f.attrs['model_config']
    if isinstance(model_config, bytes):
        model_config = model_config.decode('utf-8')
    model_config = json.loads(model_config)

    remove_time_major(model_config)

    updated_model_config = json.dumps(model_config)

    f.attrs['model_config'] = updated_model_config.encode('utf-8')

print("Successfully removed 'time_major' from the model configuration.")

