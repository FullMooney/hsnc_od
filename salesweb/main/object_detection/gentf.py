# -*- coding: utf-8 -*-

import os

def gentf(modelname):
    pyscript = 'python generate_tfrecord.py --csv_path=images/' + modelname
    os.system(pyscript)
    return 0

if __name__ == '__main__':
    gentf()