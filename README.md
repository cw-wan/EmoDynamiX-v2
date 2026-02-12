<p align="center">
  <h1 align="center"><img src="img/logo.png" alt="Logo" style="height:1em; vertical-align:middle;"> EmoDynamiX</h1>
  <h3 align="center">Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics</h3>
  <h4 align="center"><img src="img/acl-logo.png" alt="ACL Logo" style="height:1em; vertical-align:middle;"> <i>NAACL 2025 Oral</i></h4>
  <p align="center">  
    <a href="https://hal.science/hal-05466118">Paper</a>
    ·
    <a href="https://github.com/cw-wan/EmoDynamiX-v2/blob/master/Slides.pdf">Slides</a>
    ·
    <a href="https://github.com/cw-wan/EmoDynamiX-v2/blob/master/Poster.pdf">Poster*</a>
  </p>
  <span style="font-size: 60%;">* <i>Poster presented at <a href="https://coria-taln-2025.lis-lab.fr/">CORIA-TALN 2025</a>.</i></span>
</p>

![](img/architecture.jpg)

## ESC Benchmark Performance

EmoDynamiX is a strong, lightweight baseline for Emotional Support Conversation. According to a comprehensive benchmark on diverse dialog metrics ([AFlow](https://arxiv.org/abs/2602.08826), 2026), EmoDynamiX is the best in the Strategy & Knowledge model category, outperforming task- and interpretability-oriented ESC systems. It is also competitive against strong, heavier baselines like social simulation, multi-agent ESC, value RL, and proprietary LLM baselines (GPT-4o and Claude-3.5).

<img width="4068" height="2194" alt="EmoDx_benchmark_performance" src="https://github.com/user-attachments/assets/7855ce08-101e-453a-aa0d-483af9014e4f" />

## Quick Start

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
@inproceedings{wan-etal-2025-emodynamix,
  author       = {Chenwei Wan and
                  Matthieu Labeau and
                  Chlo{\'{e}} Clavel},
  editor       = {Luis Chiruzzo and
                  Alan Ritter and
                  Lu Wang},
  title        = {EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling
                  MiXed Emotions and Discourse Dynamics},
  booktitle    = {Proceedings of the 2025 Conference of the Nations of the Americas
                  Chapter of the Association for Computational Linguistics: Human Language
                  Technologies, {NAACL} 2025 - Volume 1: Long Papers, Albuquerque, New
                  Mexico, USA, April 29 - May 4, 2025},
  pages        = {1678--1695},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://doi.org/10.18653/v1/2025.naacl-long.81},
  doi          = {10.18653/V1/2025.NAACL-LONG.81},
  timestamp    = {Thu, 14 Aug 2025 11:28:41 +0200},
  biburl       = {https://dblp.org/rec/conf/naacl/WanLC25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
