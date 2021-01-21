###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
SpO2 Datasets
"""
import torch
import ai8x
import os


def spo2_get_datasets(data, load_train=True, load_test=True):

    (data_dir, args) = data
    
    train_data_dir = os.path.join(data_dir, 'SPO2', 'spo2_train_data.pt')
    test_data_dir = os.path.join(data_dir, 'SPO2', 'spo2_test_data.pt')

    if load_train:
        train_dataset = torch.load(train_data_dir)
    else:
        train_dataset = None

    if load_test:
        test_dataset = torch.load(test_data_dir)

    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'SPO2',
        'input': (1, 32, 1),
        'output': ('r'),
        'regression': True,
        'loader': spo2_get_datasets,
    },
]
