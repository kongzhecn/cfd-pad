# Fingerprint Presentation Attack Detection by Channel-wise Feature Denoising

This is an official pytorch implementation of '[Fingerprint Presentation Attack Detection by Channel-wise Feature Denoising](https://ieeexplore.ieee.org/abstract/document/9851680/)', which is accepted by *IEEE Transactions on Information Forensics and Security*.

## Requirements

- python 3.6

- pytorch 1.1.0

- torchvision 0.3.0

- numpy 1.19.5

- pandas 0.25.3

- scikit-image

  

## Pre-processing
- Dataset

  Download the LivDet 2017 datasets.

- Data Label Generation

  Move to the `$root` and generate the label:

  ```
    python data_find.py --data_path dataPath
  ```

  `dataPath` is the path of data.
  
  

## Usage

- Move to the `$root` and run:

  ```
  python train.py --save savePath
  ```

  `savePath` is the filename to save model, which is in `$root`

## Citation
Please cite our work if it's useful for your research.
```angular2html
@article{liu2022fingerprint,
  title={Fingerprint Presentation Attack Detection by Channel-Wise Feature Denoising},
  author={Liu, Feng and Kong, Zhe and Liu, Haozhe and Zhang, Wentian and Shen, Linlin},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={2963--2976},
  year={2022},
  publisher={IEEE}
}
```