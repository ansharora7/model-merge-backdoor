# Here’s a Free Lunch: Sanitizing Backdoored Models with Model Merge
This repo contains source code and pre-processed corpora for ["Here’s a Free Lunch: Sanitizing Backdoored Models with Model Merge"](https://arxiv.org/pdf/2402.19334) (accepted to Findings of ACL2024)

Authors: [Ansh Arora](https://scholar.google.com/citations?user=P-CP_R4AAAAJ&hl=en), [Xuanli He](https://xlhex.github.io/), [Maximilian Mozes](http://mmozes.net/), [Srinibas Swain](https://iiitg.irins.org/profile/139254), [Mark Dras](https://researchers.mq.edu.au/en/persons/mark-dras), [Qiongkai Xu](https://xuqiongkai.github.io/)

![](intro_fig.png)

## Abstract
> The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can significantly remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we verify our hypothesis on various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks on classification and instruction-tuned tasks without additional resources or specific knowledge. 
Our approach consistently outperforms recent advanced baselines, leading to an average of about 75\% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.

<br><br>
## Installing the required libraries
```
pip install transformers datasets torch
```

<br><br>
## Usage
### Getting the data
You can download the data used in the paper from this [link](https://drive.google.com/file/d/1bqL8mIRrnEx3AS-VcrPWD975IZ_rdpZM/view?usp=drive_link). The code relies on `.csv` files, but `.json` files are also provided (except for experiments involving the BITE backdoor attack) and can be used after minor tweaks. After downloading, unzip the data to create the `data` folder. The folder structure should be organized as follows:

```
data/
├── agnews/
   ├── agnews_badnet/
   ├── agnews_bite/
   ├── agnews_clean/
   ├── agnews_hidden/
   ├── agnews_lws/
   └── agnews_sent/
├── olid/
   ├── olid_badnet/
   ├── ...
├── qnli/
   ├── qnli_badnet/
   ├── ...
├── sst2/
   ├── sst2_badnet/
   ├── ...

```
**NOTE:** For QNLI, the `text` column in the `.csv` file can be formed by concatenating the `sent1` and `sent2` keys from the `.json` files. Representatively: `text = sent1 + " " + sent2`. 


<br><br>    
### Training a Model
We fine-tune a model on the dataset starting from a pre-trained model checkpoint from HuggingFace. More information in the [Hugging Face documentation](https://huggingface.co/docs/transformers/en/training).

You can train a model using the train.py script. Here's an example command:
```
CUDA_VISIBLE_DEVICES=1 python3 train.py --num_epochs 3 --dataset_name olid_badnet --op_dir output_dir --num_labels 2
```
**--num_epochs**: Sets the number of training epochs.

**--dataset_name**: Specifies the name of the dataset to be used for training.

**--op_dir**: Defines the directory where the output (such as model checkpoints and logs) will be saved.

**--num_labels**: Sets the number of labels/classes in the dataset.

<br><br>
### Evaluating a Model
You can evaluate a model using the test.py script. Here's an example command:
```
CUDA_VISIBLE_DEVICES=1 python3 test.py --ckpt olid_badnet/num_epochs-3/checkpoint-2484 --dataset_name olid_badnet --op_file op.txt
```
**--ckpt**: Points to the checkpoint file to load the trained model from.

**--dataset_name**: Specifies the name of the dataset to be used for testing.

**--op_file**: Defines the file where the test results will be saved.

<br><br>
### Model Merging
The code for model merging can be found in ```wag.py```. Enter the path to the checkpoints seperated by a space.
```
python3 wag.py --ckpts path_to_ckpt1 path_to_ckpts2 --save_path save.pth
```
**--ckpts**: Specifies the paths to the model checkpoint files to be used. You can list multiple checkpoints separated by a space.

**--save_path**: Defines the file where the processed results or merged model will be saved.

<br><br>
## Citation
```
@article{arora2024here,
  title={Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge},
  author={Arora, Ansh and He, Xuanli and Mozes, Maximilian and Swain, Srinibas and Dras, Mark and Xu, Qiongkai},
  journal={arXiv preprint arXiv:2402.19334},
  year={2024}
}
```
