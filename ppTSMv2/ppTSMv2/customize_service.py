import time
import numpy as np
from onnxruntime import InferenceSession
from loader import get_data
import os
from model_service.pytorch_model_service import PTServingBaseService


class fatigue_driving_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.weights = 'best.pt'
        self.capture = 'test.mp4'
        self.model = InferenceSession(model_path.replace(self.weights,'ppTSM_v2.onnx'))
    def _preprocess(self, data):
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def _inference(self, data):

        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        print(data)
        result = {"result": {"category": 0, "duration": 6000}}
        try:
            now = time.time()
            x = get_data(os.path.realpath(self.capture))
            a = self.model.run(output_names=None, input_feed={'data_batch_0': x})
            exp_arr = np.exp(a[0])
            softmax_arr = exp_arr / np.sum(exp_arr, axis=1, keepdims=True)
            r = softmax_arr.sum(axis=0)
            result['result']['category'] = int(r.argmax())
            final_time = time.time()
            result['result']['duration'] = int(np.round((final_time - now) * 1000))
        except Exception as e:
            print(e)
        finally:
            return result

    def _postprocess(self, data):
        os.remove(self.capture)
        return data
