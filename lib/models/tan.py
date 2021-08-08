import torch
from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.bert_modules as bert_modules

class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.bert_layer = getattr(bert_modules, config.TAN.VLBERT_MODULE.NAME)(config.DATASET.NAME, config.TAN.VLBERT_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, word_mask, visual_input):

        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        vis_h = vis_h.transpose(1, 2)
        logits_text, logits_visual, logits_iou, iou_mask_map = self.bert_layer(textual_input, textual_mask, word_mask, vis_h)
        # logits_text = logits_text.transpose(1, 2)
        logits_visual = logits_visual.transpose(1, 2)

        return logits_text, logits_visual, logits_iou, iou_mask_map

    def extract_features(self, textual_input, textual_mask, visual_input):
        pass
