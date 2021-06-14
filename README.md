## Postoperative brain cavity segmentation

This is the code for [Pérez-García et al., 2021, *An unsupervised learning strategy for postoperative
brain cavity segmentation simulating resections
during training* - International Journal of Computer Assisted Radiology and
Surgery (IJCARS)](https://doi.org/10.1007/s11548-021-02420-2).

If you use this code or the [EPISURG](https://github.com/fepegar/SlicerEPISURG)
dataset for your research, please cite this publication as:

> Pérez-García, F., Dorent, R., Rizzi, M. et al. A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections. Int J CARS (2021). https://doi.org/10.1007/s11548-021-02420-2

BibTeX:

```bibtex
@article{perez-garcia_self-supervised_2021,
	title = {A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections},
	issn = {1861-6429},
	url = {https://doi.org/10.1007/s11548-021-02420-2},
	doi = {10.1007/s11548-021-02420-2},
	language = {en},
	urldate = {2021-06-14},
	journal = {International Journal of Computer Assisted Radiology and Surgery},
	author = {P{\'e}rez-Garc{\'i}a, Fernando and Dorent, Reuben and Rizzi, Michele and Cardinale, Francesco and Frazzini, Valerio and Navarro, Vincent and Essert, Caroline and Ollivier, Ir{\`e}ne and Vercauteren, Tom and Sparks, Rachel and Duncan, John S. and Ourselin, S{\'e}bastien},
	month = jun,
	year = {2021},
}
```

## `resseg`

The trained models can be used to segment easily using [`resseg`](https://github.com/fepegar/resseg).

## `resector`

Resections can be simulated on 3D MRI using [`resector`](https://github.com/fepegar/resector).

## Installation

```shell
$ conda create -n ijcars python=3.7 ipython -y && conda activate ijcars
$ conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -c pytorch -y
$ pip install -r requirements.txt
```

## EPISURG dataset

See the [EPISURG extension for 3D Slicer](https://github.com/fepegar/SlicerEPISURG).

## Commands used for training

### Public

#### Augmentation

```
python main.py with config/simulated/config_simulated_no_augment.yml
```

#### Shape

##### Cuboids

```
python main.py with config/simulated/shape/config_simulated_shape_cuboid.yml
```

##### Ellipsoids

```
python main.py with config/simulated/shape/config_simulated_shape_ellipsoid.yml
```

##### Noisy ellipsoids (baseline)

```
python main.py with config/simulated/config_simulated_baseline.yml
```

#### Texture

##### Percentile 1

```
python main.py with config/simulated/texture/config_simulated_texture_dark.yml
```

##### Percentile 1, 99

```
python main.py with config/simulated/texture/config_simulated_texture_random.yml
```

##### CSF (baseline)

```
python main.py with config/simulated/config_simulated_baseline.yml
```

##### CSF + WM

```
python main.py with config/simulated/texture/config_simulated_texture_wm.yml
```

##### CSF + BC

```
python main.py with config/simulated/texture/config_simulated_texture_clot.yml
```

##### CSF + WM + BC

```
python main.py with config/simulated/texture/config_simulated_texture_clot_wm.yml
```

### Centers

#### Train

#### Load and tune

```
python main.py with config/real/config_load_train.yml with dataset_name $DATASET
```
