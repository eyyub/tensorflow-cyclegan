# tensorflow-cyclegan
A lightweight [CycleGAN](https://arxiv.org/abs/1703.10593) tensorflow implementation.

If you plan to use a CycleGAN model for real-world purposes, you should use the [Torch CycleGAN](https://github.com/junyanz/CycleGAN) implementation.

[@eyyub_s](https://twitter.com/eyyub_s)

## Some examples
![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/lion2leopard.jpg?raw=true)
lion2leopard (cherry-picked)

![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/l2l.jpg?raw=true)
More lion2leopard (each classes contain only 100 instances!)

![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/h2z_1.JPG?raw=true)

horse2zebra

![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/failure_h2z_5.JPG?raw=true)

horse2zebra failure

![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/z2h_1.JPG?raw=true)

zebra2horse

![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/weird.JPG?raw=true)

wtf

![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/h2z_epoch200.jpg?raw=true)

More zebra2horse

![](https://github.com/Eyyub/tensorflow-cyclegan/blob/master/readme_imgs/a2o.jpg?raw=true)

apple2orange

See more in `readme_imgs/`

## Build horse2zebra
- Download `horse2zebra.zip` from [CycleGAN datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)
- Unzip it here `.`
- Run `python build_dataset.py horse2zebra/trainA horse2zebra/trainB trainA trainB` 
- (make sure `dataset_trainA.npy` & `dataset_trainB.npy` are created)
- Then, run `python example.py`
- (If you want to stop and restart your training later you can do: `python example.py restore <last_iter_number>`)

## Requiremennts
- Python 3.5
- Tensorflow
- Matplotlib
- Pillow
- (Only tested on Windows so far)

## _Very_ useful info
- Training took me ~1day (GTX 1060 3g)
- Each 100 steps the script adds an image in the `images/` folder
- Each 1000 steps the model is saved in `models`
- CycleGAN seems to be init-sensitive, if the generators only inverse colors: kill & re-try training

## Todo
- [x] Image Pool
- [ ] Add learning reate linear decay
