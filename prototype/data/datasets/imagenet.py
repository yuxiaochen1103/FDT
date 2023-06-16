import pickle
from PIL import Image
from torch.utils.data import Dataset
import json
import os.path as osp
import os
import torch

class ImgnetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, split, root_dir,transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.label_2_clsname_dir = root_dir + '/val_official.json'
        self.root_dir = root_dir

        #print('imagenet dat files:', self.root_dir)
        #print('label_2_clsname_dir:', self.label_2_clsname_dir)
        assert self.label_2_clsname_dir and self.root_dir


        self.split = split

        self.fn_to_label, self.label_to_name = self.load_labelmapping()
        #load image pth list
        with open(self.root_dir + '/{}_datalist.pkl'.format(split), 'rb') as src_file:
            self.datalist = pickle.load(src_file)

        self.label_texts_ensemble = 'prompt80'

        self.transform = transform

        #print('ImageNet1K {} data num: {}'.format(split, len(self.datalist)))


    def prompt_label_name(self, label_name):
        # get prompt templates, such as: label_text = ['a photo of ' + text + '.']
        if self.label_texts_ensemble == 'prompt6':
            f = f'{osp.abspath(os.getcwd())}/prototype/data/datasets/prompts/query_pattern_prompt6'
        elif self.label_texts_ensemble == 'prompt8':
            f = f'{osp.abspath(os.getcwd())}/prototype/data/datasets/prompts/query_pattern_prompt8'
        elif self.label_texts_ensemble == 'prompt80':
            f = f'{osp.abspath(os.getcwd())}/prototype/data/datasets/prompts/query_pattern_prompt80'
        elif self.label_texts_ensemble == 'cc':
            return [label_name]
        elif 'file:' in self.label_texts_ensemble:
            f = self.label_texts_ensemble[5:]
        elif self.label_texts_ensemble == 'simple':
            f = f'{osp.abspath(os.getcwd())}/datasets/prompts/query_pattern_prompt1'
        else:
            raise NotImplementedError(self.label_texts_ensemble)
        
        # fillout templates with label names
        label_text = []
        with open(f) as fin:
            for line in fin.readlines():
                label_text.append(line.strip().replace('{0}', label_name))

        return label_text


    def load_labelmapping(self):

        #load declip's info
        fn_to_label = {}
        label_2_name = {}
        with open(self.label_2_clsname_dir) as f:
            for l in f:
                data = json.loads(l)
                #print(data)
                label = int(data['label'])
                fn = data['filename']
                label_name = data['label_name']

        
                assert data['label_name'] == data['caption'] #asser caption is label_name

                #file name to label
                fn_to_label[fn] = label

                #label to file name
                if label in label_2_name:
                    assert label_2_name[label] == label_name
                else:
                    label_2_name[label] = label_name
        #print(fn_to_label, label_2_name)
        return fn_to_label, label_2_name
            
    
    def get_label_texts(self):
        # prompt the label list
        # get label_list and itsnames
        labels = list(self.label_to_name.keys())
        labels.sort()

        label_texts = []
        label_text_len = []
        for label in labels:
            label_name = self.label_to_name[label]
            label_text = self.prompt_label_name(label_name)
            label_texts.extend(label_text)
            label_text_len.append(len(label_text))

        label_num = len(labels)
        label_texts_ensemble_matrix = torch.eye(label_num)

        # label_texts_ensemble_matrix = torch.zeros(all_len, label_num)
        # for lbl, ltl in enumerate(label_text_len):
        #     label_texts_ensemble_matrix[offset: offset + ltl, lbl] = 1
        #     offset += ltl

        return label_texts, label_texts_ensemble_matrix



    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        pth, label = self.datalist[idx]

        #print(pth)
        img_pth = self.root_dir + '/{}{}'.format(self.split, pth)

        with open(img_pth, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {'image':img, 'label':label}