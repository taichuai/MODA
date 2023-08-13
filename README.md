<div align="center">

<h2> <font color="#FF9933">M</font><font color="#66CCFF">O</font><font color="#FFCC00">D</font><font color="#33CC66">A</font>: <span style="font-size:18px"> <font color="#FF9933">M</font>apping-<font color="#66CCFF">O</font>nce Audio-driven Portrait Animation with <font color="#FFCC00">D</font>ual <font color="#33CC66">A</font>ttentions</nobr></span> </h2>

  <a href='https://arxiv.org/abs/2307.10008'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; <a href='https://liuyunfei.net/projects/iccv23-moda/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) &nbsp; [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](#) &nbsp; 

<div>
    <nobr><a href="http://liuyunfei.net"
  >Yunfei Liu</a><sup>1</sup>,</nobr>
  <nobr><a href="https://scholar.google.com.hk/citations?user=Xf5_TfcAAAAJ&hl=zh-CN"
  >Lijian Lin</a><sup>1</sup>,</nobr>
  <nobr>Fei Yu<sup>2</sup>,</nobr>
  <nobr>Changyin Zhou<sup>2</sup>,</nobr>
  <nobr><a href="https://yu-li.github.io/"
  >Yu Li</a><sup>1</sup></nobr>
 <br>
  <nobr><sup>1</sup><a href="https://www.idea.edu.cn">International Digital Economy Academy (IDEA), Shenzhen, China</a>,&nbsp;&nbsp;</nobr>
  <nobr><sup>2</sup><a href="https://www.vistring.ai">Vistring Inc., Hangzhou, China</a></nobr>
</div>
<br>
<i>—<strong><a href='https://arxiv.org/abs/2307.10008' target='_blank'>ICCV 2023</a></strong>—</i>
<br>
<br>


![moda](images/teaser.png)

<br>

</div>

## 🕊️ Description

MODA is a unified system for multi-person, diverse, and high-fidelity talking portrait generation.

## 🎊 News

1. `2023/08/13` Inference codes have been released.

MODA is a unified system for multi-person, diverse, and high-fidelity talking portrait generation.

## 🛠️ Installation

After cloning the repository please install the environment by running the `install.sh` script. It will prepare the MODA for usage. 

```shell
git clone https://github.com/DreamtaleCore/MODA.git
cd MODA
bash ./install.sh
```

## 🚀 Usage

Quick run
```shell
python inference.py
```

Parameters:
```shell
usage: Inference entrance for MODA. [-h] [--audio_fp_or_dir AUDIO_FP_OR_DIR] [--person_config PERSON_CONFIG]
                                    [--output_dir OUTPUT_DIR] [--n_sample N_SAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  --audio_fp_or_dir AUDIO_FP_OR_DIR
  --person_config PERSON_CONFIG
  --output_dir OUTPUT_DIR
  --n_sample N_SAMPLE
```

## 🚧 TODO

- [ ] Prepare the pretriained-weights
- [ ] Prepare the huggingface🤗 demo
- [ ] Release the training code
- [ ] Data preprocessing scripts

## 🛎 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{liu2023MODA,
  title={MODA: Mapping-Once Audio-driven Portrait Animation with Dual Attentions},
  author={Liu, Yunfei and Lin, Lijian and Fei, Yu and Changyin, Zhou, and Yu, Li},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```