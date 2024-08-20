# EmoDynamiX

## Download Checkpoints

|              Model               |                                                                                        URL                                                                                        |
|:--------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|       EmoDynamiX on ESConv       |                                           [link](https://drive.google.com/file/d/1BQUE-g-bjf4QOCTUdrb35EevHnD8ElQO/view?usp=drive_link)                                           |
|       EmoDynamiX on AnnoMI       |                                           [link](https://drive.google.com/file/d/1VWhx9xoC7L9roSPeP9hvXjGlyjzs-kY5/view?usp=drive_link)                                           |

Unzip to the project root directory.

## Reproduce the Results

Test on ESConv:

```shell
./test_roberta_hg_esconv.sh
```

Test on AnnoMI:

```shell
./test_roberta_hg_annomi.sh
```

## Do Your Own Training

Train on ESConv:

```shell
./train_roberta_hg_esconv.sh
```

Train on AnnoMI:

```shell
./train_roberta_hg_annomi.sh
```

## Make Your Own Data

Download weights for the submodules [here](https://drive.google.com/file/d/1KNsoWp1FjdMnrCVWiONRb6w4QUpzGuyP/view?usp=drive_link).

Run `make.py` for ESConv or AnnoMI, then run `preprocess.py`.
