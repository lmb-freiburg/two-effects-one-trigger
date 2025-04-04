from torch.nn import Module
import numpy as np

class sparsify_caption_drop_words(Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, caption):
        caption = caption.split(' ')
        drop_probs = np.random.uniform(low=0.0, high=1.0, size=(len(caption),) )        
        drop_probs = drop_probs < self.p
        caption = [cap for (cap, prob) in zip (caption, drop_probs) if prob <= self.p]
        caption = ' '.join(caption)

        return caption


class sparsify_caption_drop_continous_half(Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, caption):        
        if np.random.uniform(low=0.0, high=1.0, size=(1,) ) <= self.p:
            caption = caption.split(' ')
            drop_frist_half = np.random.uniform(low=0.0, high=1.0, size=(1,) ) <= 0.5 
            if drop_frist_half:
                caption = caption[int(len(caption)/2):]
            else:
                caption = caption[:int(len(caption)/2)]
            caption = ' '.join(caption)
        return caption
    

class sparsify_caption_drop_continous_quater(Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, caption):        
        if np.random.uniform(low=0.0, high=1.0, size=(1,) ) <= self.p:
            caption = caption.split(' ')
            rd_loc = np.random.uniform(low=0.0, high=1.0, size=(1,) )
            if rd_loc <= 0.25:
                caption = caption[:int(len(caption)/4)]
            elif rd_loc<= 0.5:
                caption = caption[int(len(caption)/4):int(len(caption)/2)]
            elif rd_loc<= 0.75:
                caption = caption[int(len(caption)/2):int(3*len(caption)/4)]
            else:
                caption = caption[int(3*len(caption)/4):]
            caption = ' '.join(caption)
        return caption    