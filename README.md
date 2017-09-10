#### Training the YOLO TL model

Details to be added...

#### Using the YOLO TL model for detection via Darkflow

1. Clone [Darkflow](https://github.com/thtrieu/darkflow)
2. Clone this repo and move contents to darkflow folder
3. Download YOLO [weights](https://drive.google.com/file/d/0B_SXDGKPsMsfYmdia1lzUjlSaXM/view?usp=sharing) into darkflow folder
4. Cythonize Darkflow (python setup.py build_ext --inplace) - works with Python 2.7 or Python 3
5. Flow all images in test_images as follows (output will go to test_images/out):

```
/flow --imgdir test_images/ --model yolo-obj.cfg --load yolo-obj_2000.weights 
```

6. Run on a single image from within Python (running on CPU):

```
(carla) MacBook-Pro:darkflow jamesjackson$ python detect.py 
/Users/jamesjackson/carnd/term3_final_project/darkflow/darkflow/dark/darknet.py:54: UserWarning: ./cfg/yolo-obj_2000.cfg not found, use yolo-obj.cfg instead
  cfg_path, FLAGS.model))
Parsing yolo-obj.cfg
Loading yolo-obj_2000.weights ...
Successfully identified 268242952 bytes
Finished in 0.0328640937805s

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 416, 416, 3)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 416, 416, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 208, 208, 32)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 208, 208, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 104, 104, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 128)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 104, 104, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 52, 52, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 256)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 52, 52, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | concat [16]                      | (?, 26, 26, 512)
 Load  |  Yep!  | local flatten 2x2                | (?, 13, 13, 2048)
 Load  |  Yep!  | concat [26, 24]                  | (?, 13, 13, 3072)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1    linear           | (?, 13, 13, 30)
-------+--------+----------------------------------+---------------
GPU mode with 1.0 usage
2017-09-10 18:03:35.175760: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-10 18:03:35.175778: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-10 18:03:35.175784: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-10 18:03:35.175789: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Finished in 7.55257201195s

1.06604599953
[{'topleft': {'y': 363, 'x': 550}, 'confidence': 0.90037251, 'bottomright': {'y': 516, 'x': 605}, 'label': 'traffic light'}]
```