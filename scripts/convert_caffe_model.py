"""
The purpose of this script is to convert pretrained weights taken from
official implementation here:
https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv2
to required format.
In a nutshell, it just renames and transposes some of the weights.
You don't have to use this script because weights are already in `src/weights`.
"""
import os
import argparse
import caffe
import numpy as np

def get_all_weights(net):
    all_weights = {}
    for p in net.params:
        if 'conv' in p:
            name = 'body.' + p
            if '-' in p:
                if '-1' in p:
                    s = 'cls.' + p
                elif '-2' in p:
                    s = 'box_offset.' + p
                else:
                    s = 'landmarks.' + p 

                all_weights[s + '.weight'] = net.params[p][0].data
                all_weights[s + '.bias'] = net.params[p][1].data
            elif len(net.params[p][0].data.shape) == 4:  # Conv layers in body
                all_weights[name + '.weight'] = net.params[p][0].data.transpose((0, 1, 3, 2))
                all_weights[name + '.bias'] = net.params[p][1].data
            else:  # Linear layer in body
                all_weights[name + '.weight'] = net.params[p][0].data
                all_weights[name + '.bias'] = net.params[p][1].data
        elif 'prelu' in p.lower(): # Prelu layers
            all_weights['body.' + p.lower() + '.weight'] = net.params[p][0].data
    return all_weights


def convert(caffe_model_folder, output_folder):
    # P-Net
    net = caffe.Net(os.path.join(caffe_model_folder, 'det1.prototxt'), os.path.join(caffe_model_folder, 'det1.caffemodel'), caffe.TEST)
    np.save(os.path.join(output_folder, 'pnet.npy'), get_all_weights(net))

    # R-Net
    net = caffe.Net(os.path.join(caffe_model_folder, 'det2.prototxt'), os.path.join(caffe_model_folder, 'det2.caffemodel'), caffe.TEST)
    np.save(os.path.join(output_folder, 'rnet.npy'), get_all_weights(net))

    # O-Net
    net = caffe.Net(os.path.join(caffe_model_folder, 'det3.prototxt'), os.path.join(caffe_model_folder, 'det3.caffemodel'), caffe.TEST)
    np.save(os.path.join(output_folder, 'onet.npy'), get_all_weights(net))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract weight from caffe model.')
    parser.add_argument('--caffe_model_folder', help='Path to caffe model (det1, det2, det3, det4)')
    parser.add_argument('--output_folder', help='Path to storing extracted weights.')
    args = parser.parse_args()
    convert(args.caffe_model_folder, args.output_folder)