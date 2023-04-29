import os
import json
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
import webdataset as wds
import random
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class CCSBUAlignDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()
        self.instructions = ["Describe this image in detail.",
                             "Take a look at this image and describe what you notice.",
                             "Please provide a detailed description of the picture.",
                             "Could you describe the contents of this image for me?"]
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]
        instruction = random.choice(self.instructions)

        return {
            "image": image,
            "instruction": instruction,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
    
class LLaVACCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))
        self.zo = ZipFile(vis_root, 'r')

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()
        
    def __getitem__(self, index):
        ann = self.annotation[index]
        image_file = ann['image']
        image = Image.open(BytesIO(self.zo.read(image_file))).convert("RGB")
        image = self.vis_processor(image)
        instruction = ann['conversations'][0]['value']
        instruction = ''.join(instruction.split('<image>')).strip()
        caption = ann["conversations"][1]['value']
        return {
            "image": image,
            "instruction": instruction,
            "text_input": caption,
            "image_id": ann['id']
        }