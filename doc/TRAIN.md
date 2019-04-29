# Train Your Own Model From Scratch

## Download the training data
For classification and bounding box regression tasks, download [WiderFace](http://shuoyang1213.me/WIDERFACE/)

For facial landmark regression task, download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

After download these two dataset, you will get file structure like this

**WIDER_FACE**
```
├── WIDER_FACE
│   ├── wider_face_split
│   ├── WIDER_test
│   ├── WIDER_train
│   └── WIDER_val
```

**CelebA**
```
├── CelebA
│   ├── Anno
│   ├── Eval
│   ├── img_align_celeba
│   ├── img_celeba
│   └── README.txt
```
Then, link these folders to "mtcnn/datasets".
```
ln -s /path/to/WIDER_FACE/* mtcnn/datasets/WIDER_FACE/
ln -s /path/to/CelebA/* mtcnn/datasets/CelebA/
```

## Train
First, we generate training data for pnet.
```bash
python scripts/gen_pnet_train.py
```
Train pnet with epoch 10, batchsize 256 and gpu:0.
```bash
python scripts/train_pnet.py -e 10 -b 256 -o output/pnet.torchm -dv cuda:0 -r True
```
Generate training data for rnet
```bash
python scripts/gen_pnet_train.py -m output/pnet.torchm
```
Train rnet
```bash
python scripts/train_rnet.py -e 10 -b 256 -o output/rnet.torchm -dv cuda:0
```
Generate training data for onet
```bash
python scripts/gen_onet_train.py -pm output/pnet.torchm -rm output/rnet.torchm
```
Train onet
```bash
python scripts/train_onet.py -e 9 -b 256 -o output/onet.torchm -dv cuda:1 -r True
```



