from networks.MIRNet_model import MIRNet

import os
import argparse

import paddle

import utils

parser = argparse.ArgumentParser(description="MIRNet_test")
parser.add_argument("--save-inference-dir", type=str, default="./output", help='path of model for export')
parser.add_argument("--model-dir", type=str, default="model_best.pdparams", help='path of model checkpoint')

opt = parser.parse_args()


def main(opt):

    model = MIRNet()
    utils.load_checkpoint(model, opt.model_dir)
    print('Loaded trained params of model successfully.')

    shape = [-1, 3, 256, 256]

    new_model = model

    new_model.eval()
    new_net = paddle.jit.to_static(
        new_model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(opt.save_inference_dir, 'model')
    paddle.jit.save(new_net, save_path)


    print(f'Model is saved in {opt.save_inference_dir}.')


if __name__ == '__main__':
    main(opt)
