# MTCNN

pytorch implementation of **inference and training stage** of face detection algorithm described in  
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

## Why this projects

[mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch) This is the most popular pytorch implementation of mtcnn. There are some disadvantages we found when using it for real-time detection task.

- No training code.
- Mix torch operation and numpy operation together, which resulting in slow inference speed.
- No unified interface for setting computation device. ('cpu' or 'gpu')
- Based on the old version of pytorch (0.2).

So we create this project and add these features:

- Add code for training stage, you can train model by your own datasets. ## TODO
- Transfer all numpy operation to torch operation, so that it can benefit from gpu acceleration. It's 10 times faster than the original repo [mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch).
- Provide unified interface to assign 'cpu' or 'gpu'.
- Based on the latest version of pytorch (1.0) and we will provide long-term support.
- It's is a component of our FaceLab ecosystem.  ## TODO
- Real-time face tracking.  ## TODO
- Friendly tutorial for beginner.

## Installation

### Pre-requests

- Anaconda3
- pytorch 0.4+
- opencv-python
- easydict
- cython

### Compile the cython code

```bash
python setup.py build_ext --inplace
```

## Test the code by example

We assume all these command running in the $SOURCE_ROOT directory.

#### Detect on example picture

```bash
python -m unittest tests.test_detection.TestDetection.test_detection
```

#### Detect on video

```bash
python scripts/detect_on_video.py --video_path ./tests/asset/video/school.avi --device cuda:0 --minsize 24
```

you can set device to 'cpu' if you have no valid gpu on your machine

## Basic Usage

```python
import cv2
import mtcnn

# First we create pnet, rnet, onet, and load weights from caffe model.
pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')

# Then we create a detector
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

# Then we can detect faces from image
img = 'tests/asset/images/office5.jpg'
boxes, landmarks = detector.detect(img)

# Then we draw bounding boxes and landmarks on image
image = cv2.imread(img)
image = mtcnn.utils.draw.draw_boxes2(image, boxes)
image = mtcnn.utils.draw.batch_draw_landmarks(image, landmarks)

# Show the result
cv2.imshwow(image)
cv2.waitKey(0)
```

## Tutorial

[Detect step by step](./tutorial/detect_step_by_step.ipynb).