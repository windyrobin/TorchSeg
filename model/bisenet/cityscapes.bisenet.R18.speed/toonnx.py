import torch
import time
import torch.onnx
from network import BiSeNet
from engine.logger import get_logger

logger = get_logger()

def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))
    #dummy_input = torch.randn(1, 3, 768, 768*2)
    #torch.onnx.export(model, dummy_input, "bisenet.onnx", verbose=True)
    #torch.onnx.export(model, dummy_input, "bisenet.onnx")
    return model

# A model class instance (class not shown)
model = BiSeNet(19, is_training=False, criterion=None, ohem_criterion=None)

# Load the weights from a file (.pth usually)
#state_dict = torch.load('./log_back/snapshot/epoch-last.pth')

# Load the weights now into a model net architecture defined by our class
#model.load_state_dict(state_dict, strict=False)

load_model(model, './log_back/snapshot/epoch-last.pth')

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(1, 3, 768, 768*2)

torch.onnx.export(model, dummy_input, "bisenet.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
