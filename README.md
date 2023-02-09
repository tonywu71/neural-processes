# conditional-neural-processes
Replication of the "Conditional Neural Processes" paper

## MNIST training
`python train.py --task mnist`

## Regression training
`python train.py --task regression`

## Celeb training
download the aligned and cropped images from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and extract to `.data`

`python train.py --task celeb`