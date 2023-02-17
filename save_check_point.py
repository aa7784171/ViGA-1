import numpy as np
import os
import zipfile
from glob import glob
from urllib.parse import quote
import sys
import socket

def save_sh_n_codes(config, ignore_dir=['']):
    os.makedirs(config["checkpoint_path"], exist_ok=True)

    name = os.path.join(config["checkpoint_path"], 'run_{}.sh'.format(socket.gethostname()))
    with open(name, 'w') as f:
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')

    name = os.path.join(config["checkpoint_path"], 'code.zip')
    with zipfile.ZipFile(name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:

        first_list = glob('*', recursive=True)
        first_list = [i for i in first_list if i not in ignore_dir]

        file_list = []
        patterns = [x + '/**' for x in first_list]
        for pattern in patterns:
            file_list.extend(glob(pattern, recursive=True))

        file_list = [x[:-1] if x[-1] == "/" else x for x in file_list]
        for filename in file_list:
            zf.write(filename)
