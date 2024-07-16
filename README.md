## Deep Orthogonal Hypersphere Compression for Anomaly Detection
This is the official implementation of Deep Orthogonal Hypersphere Compression for Anomaly Detection, ICLR 2024 (Spotlight).

### Dependencies

* python 3.8
* pytorch
* torch-geometric
* torch-sparse
* numpy
* scikit-learn

If you have installed above mentioned packages you can skip this step. Otherwise run:

    pip install -r requirements.txt

## Reproduce graph data results

The code will be available soon.


## Reproduce tabular data results

To generate results, please run:

    python demo_tabular.py

For running tabular data, the dataset name needs to be revised in corresponding demo files. 

## Reproduce image data results

To generate results, please run:

    python demo_cifar10.py


## Reference

If you find this code useful in your research, please consider citing:

```
@inproceedings{zhang2024deep,
  title={Deep Orthogonal Hypersphere Compression for Anomaly Detection},
  author={Zhang, Yunhe and Sun, Yan and Cai, Jinyu and Fan, Jicong},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
