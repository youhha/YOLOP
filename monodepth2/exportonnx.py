from __future__ import absolute_import, division, print_function
from ctypes import resize

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torchvision
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
from combine_model import Encoder_Decoder
import torch.nn as nn
import onnx
import onnxruntime as ort
import  cv2
class FullModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(FullModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs[("disp", 0)]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str, default='assets/test_image.jpg',
                        help='path to a test image or folder of images')
    parser.add_argument('--model_name', type=str, default='mono_640x192',
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            # features = encoder(input_image)
            # outputs = depth_decoder(features)

            model = Encoder_Decoder(encoder=encoder, decoder=depth_decoder)
            
            model.eval()
            outputs = model(input_image)
            x = torch.randn((1,3,192,640), device="cuda")
            output_path = "monodepth2.onnx"

            torch.onnx.export(model,               # model being run
                            x,                       # model input (or a tuple for multiple inputs)
                            output_path,   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=11,         # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['inputx'],   # the model's input names
                            output_names = ['outputy'], # the model's output names
                            verbose=True
                            )
            

            # # disp = outputs[("disp", 0)]
            # disp = outputs
            # print('disp: ', disp.shape)
            # disp_ = disp.squeeze().cpu().numpy()
            # cv2.imwrite('disp_ori.png',disp_*255)
            # disp_resized = torch.nn.functional.interpolate(
            #     disp, (original_height, original_width), mode="bilinear", align_corners=False)
                
            # # Saving colormapped depth image
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(colormapped_im)

            # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)

            # print(" Processed {:d} of {:d} images - saved predictions to:".format(
            #     idx + 1, len(paths)))
            # print("   - {}".format(name_dest_im))
            # print("   - {}".format(name_dest_npy))

    # print('-> Done!')
    # x = torch.rand(1,3,192,640)
    # input_names = ['input']
    # output_names = ['output']
    # torch.onnx.export(model, x, 'mono.onnx',input_names=input_names, output_names=output_names,opset_version=11, verbose='True')

def onnx_inference(): 
    img = cv2.imread("assets/000005.png")
    print(img.shape)
    h, w, _ = img.shape
	
	## opencv test
    blobImage = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (640, 192), None, True, False)
    net = cv2.dnn.readNet('monodepth2.onnx')
    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    outs = net.forward(outNames)
    print('cv outs: ', outs[0].shape)
    outs = np.squeeze(outs, axis=(0,1))
    outs = outs * 255.0
    outs =outs.transpose((1,2,0)).astype(np.uint8)
    disp_resized_np = cv2.resize(outs,(640,192))
    cv2.imwrite('disp_cv.png',disp_resized_np)

	## onnxruntime test 
    model = onnx.load('monodepth2.onnx')
    onnx.checker.check_model(model)
    session = ort.InferenceSession('monodepth2.onnx',providers=['CUDAExecutionProvider'])
    img = cv2.resize(img, (640, 192))
    img = np.asarray(img) / 255.0
    img = img[np.newaxis, :].astype(np.float32)

    input_image = img.transpose((0,3,1,2))
    outs = session.run(None, input_feed={'inputx':input_image})
    outs = np.squeeze(outs, axis=(0,1))
    outs = outs * 255.0
    outs =outs.transpose((1,2,0)).astype(np.uint8)
    disp_resized_np = cv2.resize(outs,(640,192))
    cv2.imwrite('disp.png',disp_resized_np)
    outs = cv2.applyColorMap(outs,colormap=cv2.COLORMAP_SUMMER)
    cv2.imwrite('disp_color.png', outs)

if __name__ == '__main__':
    # args = parse_args()
    # test_simple(args)
    onnx_inference()

