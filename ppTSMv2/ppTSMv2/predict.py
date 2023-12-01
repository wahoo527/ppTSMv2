
import time
import numpy as np
from onnxruntime import InferenceSession
from loader import get_data
model = InferenceSession('ppTSM_v2.onnx')

now = time.time()
"""在这里修改视频路径"""
video_path= '../video/day_man_052_21_1.mp4'
x = get_data(video_path)
a = model.run(output_names=None, input_feed={'data_batch_0': x})
idx = a[0].argmax(1)
unique, counts = np.unique(idx,return_counts=True)
result = {"result": {"category": 0, "duration": 6000}}
result['result']['category'] = unique[counts.argmax(0)]
final_time = time.time()
result['result']['duration'] = int(np.round((final_time - now) * 1000))
print(result)