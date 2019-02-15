import os
import glob
from PIL import Image
import numpy as np
from torch.utils import data

class WaterDataset(data.Dataset):
    '''
    Dataset for WaterDataset
    '''
    def __init__(self, root, video_name=None, mode=0):
        self.root = root
        self.mask_dir = os.path.join(root, 'annots')
        self.image_dir = os.path.join(root, 'imgs')
        self.mode = mode

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}

        if mode > 0:
            test_list_path = os.path.join(root, 'test_list.txt')
            with open(os.path.join(test_list_path), "r") as lines:
                for line in lines:
                    _video = line.rstrip('\n')
                    self.videos.append(_video)
                    self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                    _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '000000.png')).convert("P"))
                    self.num_objects[_video] = np.max(_mask)
                    self.shape[_video] = np.shape(_mask)

        else:
            _video = video_name
            self.videos.append(_video)
            self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
            _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '0.png')).convert("P"))
            self.num_objects[_video] = 1
            self.shape[_video] = np.shape(_mask)


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        if self.mode > 0:
            num_objects = self.num_objects[video]
        else:
            num_objects = 1
        info['num_objects'] = num_objects

        
        raw_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        raw_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            raw_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.

            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  #allways return first frame mask
                raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                mask_file = os.path.join(self.mask_dir, video, '00000.png')
                raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

            if self.MO:
                raw_masks[f] = raw_mask
            else:
                raw_masks[f] = (raw_mask != 0).astype(np.uint8)

            
        # make One-hot channel is object index
        oh_masks = np.zeros((self.num_frames[video],)+self.shape[video]+(num_objects,), dtype=np.uint8)
        for o in range(num_objects):
            oh_masks[:,:,:,o] = (raw_masks == (o+1)).astype(np.uint8)


        # padding size to be divide by 32
        nf, h, w, _ = oh_masks.shape
        new_h = h + 32 - h % 32
        new_w = w + 32 - w % 32
        # print(new_h, new_w)
        lh, uh = (new_h-h) / 2, (new_h-h) / 2 + (new_h-h) % 2
        lw, uw = (new_w-w) / 2, (new_w-w) / 2 + (new_w-w) % 2
        lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        pad_masks = np.pad(oh_masks, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        pad_frames = np.pad(raw_frames, ((0,0),(lh,uh),(lw,uw),(0,0)), mode='constant')
        info['pad'] = ((lh,uh),(lw,uw))

        th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(pad_frames, (3, 0, 1, 2)).copy()).float(), 0)
        th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(pad_masks, (3, 0, 1, 2)).copy()).long(), 0)
        
        return th_frames, th_masks, info