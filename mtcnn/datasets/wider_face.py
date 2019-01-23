import os
import sys
import cv2

here = os.path.dirname(__file__)


class WiderFace(object):

    def __init__(self, dataset_folder=os.path.join(here, 'WIDER_FACE')):
        """
        Make sure the WIDER_FACE dataset saved in $SOURCE_ROOT/datasets/WIDER_FACE folder.
        """
        self.dataset_folder = dataset_folder

    def _load_meta_data(self, meta_file, target_folder=''):
        """Load metadata of WIDER_FACE dataset.

        Args:
            meta_file (str): E.g. WIDER_FACE/wider_face_split
            target_folder (str): E.g. WIDER_FACE/WIDER_train/images

        Returns:
            list: Each item contains a dict with file_name, num_bb (Number of bounding box), meta_data(x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose).
        """
        f = open(meta_file)

        ret = []

        flag = 0
        num = 0
        current_num = 0
        current_dict = {}

        for line in f:
            if flag == 0:
                current_dict = {'file_name': os.path.join(
                    target_folder, line.strip())}
                flag = 1

            elif flag == 1:
                current_dict['num_bb'] = int(line.strip())
                num = int(line.strip())
                current_dict['meta_data'] = list()
                flag = 2

            elif flag == 2:
                cur = [int(i) for i in line.strip().split(' ')]

                # Append the boxes whoes attribute 'invalid' is 'True'.
                if cur[7] == 0:
                    current_dict['meta_data'].append(cur)
                else:
                    current_dict['num_bb'] -= 1
                current_num += 1

                if current_num == num:
                    ret.append(current_dict)
                    flag = 0
                    current_num = 0

        f.close()

        return ret

    def get_train_meta(self):
        meta_file = os.path.join(self.dataset_folder, "wider_face_split/wider_face_train_bbx_gt.txt")
        target_folder = os.path.join(self.dataset_folder, 'WIDER_train/images')

        return self._load_meta_data(meta_file, target_folder)

    def get_val_meta(self):
        meta_file = os.path.join(self.dataset_folder, "wider_face_split/wider_face_val_bbx_gt.txt")
        target_folder = os.path.join(self.dataset_folder, 'WIDER_val/images')

        return self._load_meta_data(meta_file, target_folder)


    def get_test_meta(self):
        """Use for load test meta_file without label.

        Returns:
            list: Each item is a file name
        """
        meta_file = os.path.join(self.dataset_folder, 'wider_face_split/wider_face_test_filelist.txt')
        target_folder = os.path.join(self.dataset_folder, 'WIDER_test/images')
        f = open(meta_file)
        ret = list()

        for line in f:
            ret.append(os.path.join(target_folder, line.strip()))

        f.close()
        return ret
