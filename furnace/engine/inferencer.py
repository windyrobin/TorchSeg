import os
import cv2
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

from engine.logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir
from utils.img_utils import pad_image_to_shape, normalize

import tensorrt as trt
import example
import common
import pycuda.driver as cuda
import pycuda.autoinit

logger = get_logger()


class Inferencer(object):
    def __init__(self, dataset, class_num, image_mean, image_std, network,
                 multi_scales, is_flip, devices,
                 verbose=False, save_path=None, show_image=False, use_trt=False):
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.class_num = class_num
        self.image_mean = image_mean
        self.image_std = image_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.context = mp.get_context('spawn')
        self.val_func = None
        #self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image

        self.iter = 0;
        self.use_trt=use_trt

    def run(self, model_path, model_indice, log_file, log_file_link):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]

            models = os.listdir(model_path)
            models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)

            for idx, m in enumerate(models):
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in
                      model_slice]
        else:
            models = [os.path.join(model_path,
                                   'epoch-%s.pth' % model_indice), ]

        results = open(log_file, 'a')
        link_file(log_file, log_file_link)

        if self.use_trt:
            self.trt_engine = self.get_trt_engine('./bisenet-half.trt')
            self.single_process();

        else:
           for model in models:
               logger.info("Load Model: %s" % model)
               self.val_func = load_model(self.network, model)
               self.multi_process_evaluation()

               results.write('Model: ' + model + '\n')
               results.write('\n')
               results.flush()

           results.close()

    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):
            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info(
                'GPU %s handle %d data.' % (device, len(shred_list)))
            p = self.context.Process(target=self.worker,
                                     args=(shred_list, device))
            procs.append(p)

        for p in procs:
            p.start()


        for p in procs:
            p.join()

        logger.info(
            'Inference Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))

    def worker(self, shred_list, device):
        start_load_time = time.time()
        logger.info('Load Model on Device %d: %.2fs' % (
            device, time.time() - start_load_time))
        for idx in shred_list:
            dd = self.dataset[idx]
            self.func_per_iteration(dd, device)

    def single_process(self):
        for dd in self.dataset:
            self.func_per_iteration(dd, self.devices[0])
    
    def func_per_iteration(self, data, device):
        raise NotImplementedError

    def compute_metric(self, results):
        raise NotImplementedError

    # evaluate the whole image at once
    def whole_eval(self, img, output_size, device=None):
        processed_pred = np.zeros(
            (output_size[0], output_size[1], self.class_num))

        for s in self.multi_scales:
            scaled_img = cv2.resize(img, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_LINEAR)
            scaled_img = self.process_image(scaled_img, None)
            #t_img = scaled_img.transpose(2, 0, 1)
            if self.use_trt:
                pred = self.val_func_process_trt(scaled_img, device)
            else:
                pred = self.val_func_process(scaled_img, device)
            
            #multi-scale not supported now
            #pred = pred.permute(1, 2, 0)
            #processed_pred += cv2.resize(pred.cpu().numpy(),
                                         #(output_size[1], output_size[0]),
                                         #interpolation=cv2.INTER_LINEAR)

        print('shape of pred', pred.shape)
        #pred = processed_pred.argmax(2)
        #np.savetxt('pred.txt', pred)

        return pred.cpu().numpy()

    # slide the window to evaluate the image
    def sliding_eval(self, img, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process(img_scale,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process(self, img, ori_shape, crop_size, stride_rate,
                      device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, margin = self.process_image(img, crop_size)
            score = self.val_func_process(input_data, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def val_func_process(self, input_data, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        print("inputs:")
        print('input shape:', input_data.shape)
        self.print_statics(input_data)
        input_data = torch.FloatTensor(input_data).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                scores = self.val_func(input_data)
                score = scores[0]

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    score_flip = self.val_func(input_data)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                #score = torch.exp(score)
                # score = score.data
        for i in range(1):
            arr = scores[i].cpu().numpy();
            print('\noutput arr shape:', arr.shape)
            #print('output arr size:', arr.shape[0]*arr.shape[2]*arr.shape[3])
            self.print_statics(arr)
        return score

    def get_engine(self, engine_file_path, plugin_factory, TRT_LOGGER):
        import tensorrt as trt
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            #with open(engine_file_path, "rb") as f, trt4.infer.create_infer_runtime(G_LOGGER) as runtime:
                #return runtime.deserialize_cuda_engine(f.read())
                print(plugin_factory)
                return runtime.deserialize_cuda_engine(f.read(), plugin_factory)
                #return runtime.deserialize_cuda_engine(f.read(), plugin_factory)

#plugin_factory = trt.OnnxPluginFactory(TRT_LOGGER)

    def get_trt_engine(self, engine_file):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        plugin_factory= example.Create(TRT_LOGGER)
        engine = self.get_engine(engine_file, plugin_factory, TRT_LOGGER) 
        context = engine.create_execution_context()                                                                                             
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        return {
            'logger': TRT_LOGGER,
            'inputs' : inputs,
            'outputs' : outputs,
            'bindings': bindings,
            'stream' : stream,
            'engine': engine,
            'context': context,
            'plugin_factory': plugin_factory
        }

    def print_statics(self, arr):
        mean = np.mean(arr)
        max = np.max(arr)
        min = np.min(arr)
        std = np.std(arr)
        print('max,min,mean,std:', max, min, mean, std)
        
    def val_func_process_trt(self, input_data, device=None):
        #input_data = np.ascontiguousarray(input_data[None, :, :, :],
        #                                  dtype=np.float32)

        input_data = np.array(input_data, dtype=np.float32, order='C') 
        #np.save('np_' + str(self.iter) + '.npy', input_data)
        #input_data.tofile('np_' + str(self.iter) + '.bin')
        #
        #self.iter = self.iter + 1 

        print('input image:')
        print('input shape:', input_data.shape)
        self.print_statics(input_data);
        
        self.trt_engine['inputs'][0].host = input_data
        begin_time = time.time()
        trt_outputs = common.do_inference(self.trt_engine['context'], bindings=self.trt_engine['bindings'], inputs=self.trt_engine['inputs'], outputs=self.trt_engine['outputs'], stream=self.trt_engine['stream'])
        end_time = time.time()
        print('infer cost:', round(1000*(end_time -begin_time)))

        #for i in range(6):
        for i in range(1):
            arr = trt_outputs[i]
            print('\nshape:', len(arr))
            self.print_statics(arr)

        score = trt_outputs[-1].reshape ((768, 768*2))

        score = torch.FloatTensor(score)
        #score = torch.exp(score)
        return score
    

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img
