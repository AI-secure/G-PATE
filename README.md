# G-PATE

This is the official code base for our NeurIPS 2021 paper:

["G-PATE: Scalable Differentially Private Data Generator via Private Aggregation of Teacher Discriminators."](https://arxiv.org/abs/1906.09338)

Yunhui Long*, Boxin Wang*, Zhuolin Yang, Bhavya Kailkhura, Aston Zhang, Carl A. Gunter, Bo Li

## Citation
```
@article{long2021gpate,
  title={G-PATE: Scalable Differentially Private Data Generator via Private Aggregation of Teacher Discriminators},
  author={Long, Yunhui and Wang, Boxin and Yang, Zhuolin and Kailkhura, Bhavya and Zhang, Aston and Gunter, Carl A. and Li, Bo},
  journal={NeurIPS 2021},
  year={2021}
}
```

## Usage

### Prepare your environment

Download required packages

```shell script
pip install -r requirements.txt
```

### Prepare your data

Please store the training data in `$data_dir`. By default, `$data_dir` is set to `../../data`.

We provide a script to download the MNIST and Fashion Mnist datasets. 

```shell script
python download.py [dataset_name]
```

For MNIST, you can run 

```shell script
python download.py mnist
```

For Fashion-MNIST, you can run 

```shell script
python download.py fashion_mnist
```

For CelebA datasets, please refer to their official websites for downloading. 

### Training 

```shell script
python main.py --checkpoint_dir [checkpoint_dir] --dataset [dataset_name] --train
```

Example of one of our best commands on MNIST:

Given eps=1,
```shell script
python main.py --checkpoint_dir mnist_teacher_4000_z_dim_50_c_1e-4/ --teachers_batch 40 --batch_teachers 100 --dataset mnist --train --sigma_thresh 3000 --sigma 1000 --step_size 1e-4 --max_eps 1 --nopretrain --z_dim 50 --batch_size 64
```

By default, after it reaches the max epsilon=1, it will generate 100,000 DP samples as `eps-1.00.data.pkl` in `checkpoint_dir`.


Given eps=10,
```shell script
python main.py --checkpoint_dir mnist_teacher_2000_z_dim_100_eps_10/ --teachers_batch 40 --batch_teachers 50 --dataset mnist --train --sigma_thresh 600 --sigma 100 --step_size 1e-4 --max_eps 10 --nopretrain --z_dim 100 --batch_size 64
```

By default, after it reaches the max epsilon=10, it will generate 100,000 DP samples as `eps-9.9x.data.pkl` in `checkpoint_dir`.

# Generating synthetic samples

python main.py --checkpoint_dir [checkpoint_dir] --dataset [dataset_name]

## Evaluate the synthetic records

We follow the standard the protocl and train a classifier on synthetic samples and test it on real samples.

For MNIST,
```shell script
python evaluation/train-classifier-mnist.py --data [DP_data_dir]
```

For Fashion-MNIST,
```shell script
python evaluation/train-classifier-fmnist.py --data [DP_data_dir]
```

For CelebA-Gender,
```shell script
python evaluation/train-classifier-celebA.py --data [DP_data_dir]
```

For CelebA-Gender (Small),
```shell script
python evaluation/train-classifier-small-celebA.py --data [DP_data_dir]
```


For CelebA-Hair,
```shell script
python evaluation/train-classifier-hair.py --data [DP_data_dir]
```

The `[DP_data_dir]` is where your generated DP samples are located. 

In the MNIST example above, we have generated DP samples in `$checkpoint_dir/eps-1.00.data`.

During evaluation, you should run with DP_data_dir=`$checkpoint_dir/eps-1.00.data`.

```shell script
python evaluation/train-classifier-mnist.py --data $checkpoint_dir/eps-1.00.data
```