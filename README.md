<h1 style="text-align: center;">EmoDynamiX</h1>

Official repository of our <img src="img/acl-logo.png" alt="ACL Logo" style="height:1em; vertical-align:middle;"> **NAACL 2025 main** conference paper:

[EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics](https://arxiv.org/abs/2408.08782).

![](img/architecture.jpg)

## Usage

[DEMO.ipynb](DEMO.ipynb) shows an example of using EmoDynamiX. You could integrate EmoDynamiX with any LLM you like to make your own strategy-controlled ESC agent!

## Download Checkpoints

|              Model               |                                                                                        URL                                                                                        |
|:--------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|       EmoDynamiX on ESConv       |                                           [link](https://drive.google.com/file/d/1pbBH5pbw5bY-35avobkdzqi0gv_bL_pn/view?usp=drive_link)                                           |
|       EmoDynamiX on AnnoMI       |                                           [link](https://drive.google.com/file/d/1VWhx9xoC7L9roSPeP9hvXjGlyjzs-kY5/view?usp=drive_link)                                           |
| Pretrained Submodules | [link](https://drive.google.com/file/d/1KNsoWp1FjdMnrCVWiONRb6w4QUpzGuyP/view?usp=drive_link) |

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

## Training from Scratch

Train on ESConv:

```shell
./train_roberta_hg_esconv.sh
```

Train on AnnoMI:

```shell
./train_roberta_hg_annomi.sh
```

## Citation

If you find our work useful, please cite our paper:

```bibtex
@misc{wan2024emodynamixemotionalsupportdialogue,
      title={EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics}, 
      author={Chenwei Wan and Matthieu Labeau and Chloé Clavel},
      year={2024},
      eprint={2408.08782},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.08782}, 
}
```
