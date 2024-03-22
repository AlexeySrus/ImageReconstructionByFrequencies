from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import os


class EvaluateLogger(object):
    def __init__(self, file_path: str, annotation: str = ''):
        _, ext = os.path.splitext(file_path)
        assert ext.upper() == '.CSV', 'Table must me an CSV file'

        self.file_path = file_path
        self.version = annotation

        self.data: Dict[str, List[Union[int, float]]] = {
            'H': [],
            'V': [],
            'D': [],
            'lvl': [],
            'version': []

        }

    def __call__(self, h_loss: float, v_loss: float, d_loss: float, level: int):
        self.data['H'].append(h_loss)
        self.data['V'].append(v_loss)
        self.data['D'].append(d_loss)
        self.data['lvl'].append(level)
        self.data['version'].append(self.version)

    def save_file(self):
        csv_data = pd.DataFrame(self.data)
        csv_data.to_csv(self.file_path, mode='a', header=not os.path.exists(self.file_path), index=False)
