import json
import torch
from PIL import Image
from easydict import EasyDict


class COCO_Dataset(torch.utils.data.Dataset):

    def __init__(self, base_fold, transform=None, is_train=False):



        if not is_train:
            self.img_fold = base_fold + '/val2014'
            self.data_file_pth = base_fold + '/testall.pkl' # a list of dict, each dict has imgname, imgid, and 5 captions
        else:
            self.img_fold = base_fold + '/train2014'
            self.data_file_pth = base_fold + '/train.pkl'


        self.transform = transform
        self.is_train = is_train
        self.load_data()
        print('load {} image names, {} captions, and {} image-text pair is_train: {}'.format(len(self.imgname_list), len(self.allcap),
                                                                                                len(self.imgtxt_list), is_train))
        assert len(self.imgname_list) * 5 == len(self.allcap)
        
        if self.is_train:
            assert len(self.imgtxt_list) == len(self.allcap)
        else:
            assert len(self.imgtxt_list) == len(self.imgname_list)


    def load_data(self):

        with open(self.data_file_pth) as f:
            data = json.load(f) # a json list, each element has image_id, image_name, and 5 captions #see data.preprocess.coco.py

        self.imgtxt_list = []
        self.imgname_list = []
        self.allcap = []

        if not self.is_train:
            # for validation, we do not need to repeat the image for 5 times, and only select the first captions
            #the capion in imgtxt_list will not feed to text encoder
            for ele in data:
                self.imgtxt_list.append((ele['imgname'], ele['cap_list'][0])) #all cap
                self.imgname_list.append(ele['imgname'])
                self.allcap += ele['cap_list']
        else:
            for ele in data:
                self.imgname_list.append(ele['imgname'])
                for cap in ele['cap_list']:
                    self.imgtxt_list.append((ele['imgname'], cap))
                    self.allcap.append(cap)
    
    def __len__(self):
        return len(self.imgtxt_list)

    def __getitem__(self, index):

        imgname, cap = self.imgtxt_list[index]

        imgpth = self.img_fold + '/' + imgname
        img = Image.open(imgpth).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        

        output = {'image': img,
                    'index': index,
                    'caption': cap,
                    'imgname': imgname}
        return output

    def collect_fn(self, data):
        images = []
        captions = []
        indexs = []
        imgnames = []

        for _, ibatch in enumerate(data):
            images.append(ibatch["image"])
            captions.append(ibatch['caption'])
            indexs.append(ibatch['index'])
            imgnames.append(ibatch['imgname'])

        #print(indexs)
        images = torch.stack(images, dim=0)
        indexs = torch.tensor(indexs)

        res = EasyDict(
                {"images": images,
               "captions": captions,
               'indexs': indexs,
               'imgnames':imgnames, 
               })

        return res   
