from torch.utils.data import DataLoader, Dataset
import jsonlines
import os
class RelationReasoningDataset(Dataset):
    def __init__(self, data_path, image_folder_path, prompt, limits=None, prompt_suffix = ''):
        if type(data_path) == str:
            self.dataset = list(jsonlines.open(data_path))
            self.image_folder_path = image_folder_path
        else:
            self.dataset = []
            for i, dp in enumerate(data_path):
                
                _dataset =  list(jsonlines.open(dp))
                for d in _dataset:
                    d['image'] = os.path.join(image_folder_path[i], d['image'])
                self.dataset+= _dataset
            self.image_folder_path = ''
        if limits:
            self.dataset = self.dataset[:limits]
        
        self.prompt = prompt
        self.prompt_suffix = prompt_suffix

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        question = data["question"]
        answer = str(data["answer"])
        image =  os.path.join(self.image_folder_path, data["image"])
        
        returns =  {"message": 
                    [
                        {
                            "role": "user", 
                            "content": [
                                    {"type": "image", "image": image}, 
                                    {"type": "text", "text": self.prompt+ "\nQuestion: " + question + self.prompt_suffix} 
                                ]
                        } 
                    ], 
                "gt_answer": answer, 
                "image": image,
                "dataset": data["dataset"],
                }
        if 'key_words' in data:
            returns['key_words'] = data['key_words']
        if 'bboxs' in data:
            returns['bboxs'] = data['bboxs']
        if 'width' in data:
            returns['width'] = data['width']
        if 'height' in data:
            returns['height'] = data['height']
        
     
        return returns
        
        
