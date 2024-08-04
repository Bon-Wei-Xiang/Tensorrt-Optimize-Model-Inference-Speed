import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2


class HostDeviceMemory:
    """
    :parameter:
    host_mem : numpy array (pagelocked memory(pinned memory)) use for put predict data here (RAM)
    device_mem : a pointer is point to gpu memory
    """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Tensorrt_engine:
    def __init__(self, engine_path):
        self.Logger = trt.Logger(trt.Logger.INFO)
        self.Engine = self.__load_engine(engine_path=engine_path)
        self.context = self.Engine.create_execution_context()
        self.Stream = cuda.Stream()
        self.__binding, self.__input_layer, self.__output_layer = self.__allocate_buffer()

    def __load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(logger=self.Logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    # allocate all buffer and gpu memory for each layer from engine
    def __allocate_buffer(self):
        """
        :return:
        binding : a list which is store all the pointer to gpu memory [device_mem, device_mem]
        input_layer : a list which is store all the input layer example : [object(host_mem, device_mem), ...]
        output_layer : a list which is store all the output layer example : [object(host_mem, device_mem), ...]
        """
        binding, input_layer, output_layer = [], [], []
        for i in self.Engine:
            layer_shape = self.Engine.get_tensor_shape(i)
            layer_type = trt.nptype(self.Engine.get_tensor_dtype(i))
            sample = np.empty(layer_shape, layer_type)
            host_mem = cuda.pagelocked_empty_like(sample)  # create an array which is pagelock(pinned memory) in RAM
            device_mem = cuda.mem_alloc_like(host_mem)  # allocate GPU memory
            binding.append(int(device_mem))
            if self.Engine.binding_is_input(i):
                input_layer.append(HostDeviceMemory(host_mem=host_mem, device_mem=device_mem))
            else:
                output_layer.append(HostDeviceMemory(host_mem=host_mem, device_mem=device_mem))
        return binding, input_layer, output_layer

    # inference
    def __call__(self, img):
        """
        :param  img: numpy(array)
        :return
        prediction result [object ness, box x, box y, box h, box w, class1 accuracy,class2 accuracy ,class3 accuracy]
        """
        # copy image to pagelock array
        for i in self.__input_layer:
            i.host[:] = img[:]

        # copy image from pagelock array to gpu memory
        [cuda.memcpy_htod_async(i.device, i.host, self.Stream) for i in self.__input_layer]
        # tensorrt execute the prediction task
        self.context.execute_async_v2(bindings=self.__binding, stream_handle=self.Stream.handle)
        # copy prediction result from gpu memory to pagelock array
        [cuda.memcpy_dtoh_async(i.host, i.device, self.Stream) for i in self.__output_layer]
        # synchronize the stream make sure the inference has return back to cpu
        self.Stream.synchronize()
        return [i.host for i in self.__output_layer]


def image_preprocess(img):
    img = np.copy(img)[:, :, ::-1]  # rotate color channel BGR to RGB
    img = letterbox(img).transpose(2, 0, 1).astype(np.float32) / 255.
    img = np.ascontiguousarray(img)
    return img[None]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    # add grid to img
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im


"""
    note !!!!!!!!!!
    below postprocess function just only  yolov5 model  
"""

def intersection_over_union(boxes, scores, iou_threshold):
    """
    :param boxes: np.array([[x1, y1, x2, y2], ...])
    :param scores: np.array([[acc], ...])
    :param iou_threshold: take which box overlap is less than iou_threshold
    :return: index for low box overlap
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []

    index = scores.argsort()[::-1]  # rotate(big to small)

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= iou_threshold)[0]

        index = index[idx + 1]  # because index start from 1

    return keep


def non_max_suppression(prediction, conf_threshold=0.25, iou_thres=0.45, max_detection_object=1000):
    # input : [x,y,w,h, object_ness, cls_acc, ..]
    # return : [x,y,x,y, acc, cls]
    objects = prediction[np.nonzero(prediction[:, 4] >= conf_threshold)]  # filter low objectness box
    max_wh = 7680  # (area) maximum box width and height

    if not objects.shape[0]:
        return None

    # conf = obj_conf * cls_conf
    objects[:, 5:] *= objects[:, 4:5]
    box = xywh2xyxy(objects[:, :4])

    high_acc, high_acc_index = np.max(objects[:, 5:], axis=1, keepdims=True), np.argmax(objects[:, 5:], axis=1,
                                                                                        keepdims=True)

    objects = np.concatenate((box, high_acc, high_acc_index.astype(np.float32)), axis=1)[
        np.nonzero(high_acc > conf_threshold)[0]]
    c = objects[:, 5:6] * max_wh  # Use class accuracy to get offset
    boxes, scores = objects[:, :4] + c, objects[:, 4]  # boxes (offset by class), scores
    i = intersection_over_union(boxes, scores, iou_thres)  # NMS
    if len(i) > max_detection_object:  # limit detections
        i = i[:max_detection_object]

    return objects[i]


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = (x[:, 0] - (x[:, 2] / 2))  # top left x
    y[:, 1] = (x[:, 1] - (x[:, 3] / 2))  # top left y
    y[:, 2] = (x[:, 0] + (x[:, 2] / 2))  # bottom right x
    y[:, 3] = (x[:, 1] + (x[:, 3] / 2))  # bottom right y
    return y


def scale_coords(new_image_shape, coords, original_image_shape):
    # input [x,y,x,y, acc, class]
    # Rescale coords (xyxy) from resize_img to original img
    gain = min(new_image_shape[0] / original_image_shape[0],
               new_image_shape[1] / original_image_shape[1])  # gain  = old / new
    pad = (new_image_shape[1] - original_image_shape[1] * gain) / 2, (
            new_image_shape[0] - original_image_shape[0] * gain) / 2  # wh padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, 0].clip(0, original_image_shape[1])  # x1
    coords[:, 1].clip(0, original_image_shape[0])  # y1
    coords[:, 2].clip(0, original_image_shape[1])  # x2
    coords[:, 3].clip(0, original_image_shape[0])  # y2

    return coords
