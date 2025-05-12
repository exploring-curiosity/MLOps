import torch, timm, numpy as np
import torch.nn as nn
from peft import get_peft_model, LoraConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "/workspace/tmp_stream/best_effb3_lora.pt"
CLASSES = sorted([line.strip() for line in open("/workspace/tmp_stream/classes.txt")])
TARGET_MODULES = ["conv_pw","conv_dw","conv_pwl","conv_head"]
MODULES_TO_SAVE = ["classifier"]

def build_model(num_classes):
    base = timm.create_model("efficientnet_b3", pretrained=True)
    base.conv_stem = nn.Conv2d(1, base.conv_stem.out_channels, kernel_size=base.conv_stem.kernel_size,
                                stride=base.conv_stem.stride, padding=base.conv_stem.padding, bias=False)
    base.classifier = nn.Linear(base.classifier.in_features, num_classes)
    lora_cfg = LoraConfig(r=12, lora_alpha=24, lora_dropout=0.1, target_modules=TARGET_MODULES,
                          modules_to_save=MODULES_TO_SAVE, bias="none", task_type="FEATURE_EXTRACTION",
                          inference_mode=True)
    return get_peft_model(base, lora_cfg)

def load_model():
    model = build_model(len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    return model, CLASSES

def predict(model, mel_array):
    x = torch.tensor(mel_array).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)[0])
    return probs.cpu()
