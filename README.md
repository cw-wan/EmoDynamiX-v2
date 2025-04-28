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
@inproceedings{wan-etal-2025-emodynamix,
    title = "{E}mo{D}ynami{X}: Emotional Support Dialogue Strategy Prediction by Modelling {M}i{X}ed Emotions and Discourse Dynamics",
    author = "Wan, Chenwei  and
      Labeau, Matthieu  and
      Clavel, Chlo{\'e}",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.81/",
    pages = "1678--1695",
    ISBN = "979-8-89176-189-6",
    abstract = "Designing emotionally intelligent conversational systems to provide comfort and advice to people experiencing distress is a compelling area of research. Recently, with advancements in large language models (LLMs), end-to-end dialogue agents without explicit strategy prediction steps have become prevalent. However, implicit strategy planning lacks transparency, and recent studies show that LLMs' inherent preference bias towards certain socio-emotional strategies hinders the delivery of high-quality emotional support. To address this challenge, we propose decoupling strategy prediction from language generation, and introduce a novel dialogue strategy prediction framework, EmoDynamiX, which models the discourse dynamics between user fine-grained emotions and system strategies using a heterogeneous graph for better performance and transparency. Experimental results on two ESC datasets show EmoDynamiX outperforms previous state-of-the-art methods with a significant margin (better proficiency and lower preference bias). Our approach also exhibits better transparency by allowing backtracing of decision making."
}
```
