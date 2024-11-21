#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from transformers import BertTokenizer

# import helpers.tokenization as tokenization
# import helpers.data_processing as dp


class BertCalibrator(trt.IInt8LegacyCalibrator):
    #实现了一个 TensorRT 量化校准器（BertCalibrator） 类，用于通过校准过程生成适合于 INT8 精度推理的量化参数缓存。下面是对这段代码的详细解析
    #BertCalibrator 类继承自 trt.IInt8LegacyCalibrator，是一个用于在 INT8 模式下执行模型推理的校准器。
    #INT8精度推理通常能够提高推理速度，并减少内存消耗，但是在进行INT8推理前，TensorRT需要进行“量化校准”来确定哪些参数可以被安全地转换为INT8
    def __init__(self, data_txt, bert_path, cache_file, batch_size, max_seq_length, num_inputs):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)

        tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.input_ids_list = []
        self.token_type_ids_list = []
        self.position_ids_list = []
        
        # TODO: your code, read inputs

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        if num_inputs > len(self.input_ids_list):
            self.num_inputs = len(self.input_ids_list)
        else:
            self.num_inputs = num_inputs
        self.doc_stride = 128
        self.max_query_length = 64

        # Allocate enough memory for a whole batch.
        self.device_inputs = [cuda.mem_alloc(self.max_seq_length * trt.int32.itemsize * self.batch_size) for binding in range(3)]

    def free(self):#释放之前分配的 GPU 内存。确保在完成校准后释放资源，防止内存泄漏
        for dinput in self.device_inputs:
            dinput.free()

    def get_batch_size(self):#返回批次大小。此方法在 TensorRT 引擎需要获取批次大小时被调用
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):#获取一批输入数据并将其准备好传送到 GPU 进行校准
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(current_batch, self.batch_size))

        # TODO your code, copy input from cpu to gpu

        return self.device_inputs

    def read_calibration_cache(self):#如果指定的缓存文件存在，读取并返回缓存的数据。如果没有缓存文件，则返回 None，表示需要重新执行校准
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):#生成的校准缓存数据写入文件。这一步是校准过程的关键，缓存可以在后续使用中复用，而无需每次都重新执行校准
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):#返回量化的分位数（即百分位）。在量化过程中，TensorRT会基于给定的分位数来选择最佳的量化范围。0.9999表示只考虑99.99%的数据范围
        return 0.9999

    def get_regression_cutoff(self):#返回回归的截止值，量化过程中用于选择最佳量化边界
        return 1.0

    def read_histogram_cache(self, length):#这两个方法用于处理直方图缓存数据。在这个实现中，它们的功能是空的，可能是为了保留未来扩展的空间。直方图缓存是TensorRT量化过程中的一部分
        return None

    def write_histogram_cache(self, ptr, length):
        return None

if __name__ == '__main__':
    data_txt = "calibrator_data.txt"
    bert_path = "bert-base-uncased"
    cache_file = "bert_calibrator.cache"
    batch_size = 1
    max_seq_length = 200
    num_inputs = 100
    cal = BertCalibrator(data_txt, bert_path, cache_file, batch_size, max_seq_length, num_inputs)

    cal.get_batch("input")
    cal.get_batch("input")
    cal.get_batch("input")
