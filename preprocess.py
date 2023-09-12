# import necessary dependencies

import pandas as pd
import numpy as np
from tensorflow import image
from tensorflow.keras.utils import to_categorical

def get_interesting_idx():
    """"
    Return a set of interesting index to extract
    """
    lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291][::2][1:]
    lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291][::2][1:]
    lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308][::2][1:]
    lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308][::2][1:]

    LIPS = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
    pose = [489, 490, 492, 493, 494, 498, 499, 500, 501, 502, 503, 504, 505, 506,507, 508, 509, 510, 511, 512]

    l_hand = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,481, 482, 483, 484, 485, 486, 487, 488]
    r_hand =[522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

    interesting_idx = np.array(LIPS + l_hand + pose + r_hand)
    return interesting_idx


class load_data:
    def __init__(self,idx,shape=(160,80,3)):
        self.idx = idx
        self.shape = shape
        self.ROWS_PER_FRAME = 543
        self.data_columns = ["x", "y", "z"]

    def load_relavent_data(self,source,num_classes,sep_by,method='nearest'):
        """
        Args:
            source (str): should be path combined with sign 
            num_classes (int) : total classes
            sep_by (str) : symbol seperates path and label
            method (str) : interpolate method (default = nearest) ref tf.image.resize methods

        Returns:
            Tensor: return a 3D(160,80,3) tensor along with categorical enoded labels
        """
        source = source.split(sep_by)
        path = source[0]
        sign = int(source[1])
        data = pd.read_parquet(path, columns=self.data_columns)
        self.n_frames = int(len(data) / self.ROWS_PER_FRAME)
        data = data.values.reshape(self.n_frames, self.ROWS_PER_FRAME, len(self.data_columns))
        data = np.where(np.isnan(data),0,data)
        data = data[:,self.idx,:]
        data = image.resize(data,self.shape[:2],method=method)
        label = to_categorical(sign,num_classes=num_classes)
        self.data = data
        return self.data,label
    
    def load_no_sign_data(self,array,method='nearest'):
        """
        Args:
            array (numpy): 3D array of landmarks datapoints
            method (str) : interpolate method (default = nearest) ref tf.image.resize methods

        Returns:
            array: returns a array shape (160,80,3) with interesting datapoints extracted
        """
        array = array[:,self.idx,:]
        array = image.resize(array,self.shape[:2],method=method)
        return array
    
    

    

            