from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import base64
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
import timm
from transformers import ViTForImageClassification, ViTConfig
import time
import io
import os
import gc

app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Device & MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Available models
MODEL_PATHS = {
    "ViT": ("vit", "models/vit_fine_tuned.pth"),
    "ResNet50": ("resnet50", "models/resnet50_fine_tuned.pth"),
    "VGG16": ("vgg16", "models/vgg16_deepfake_model.pth"),
    "DenseNet121": ("densenet121", "models/densenet121_fine_tuned.pth"),
    "EfficientNet-B0": ("efficientnet_b0", "models/efficientnet_b0_fine_tuned.pth"),
    "EfficientNet-B1": ("efficientnet_b1", "models/efficientnet_b1_fine_tuned.pth"),
    "EfficientNet-B4": ("efficientnet_b4", "models/efficientnet_b4_fine_tuned.pth"),
    "EfficientNet-B5": ("efficientnet_b5", "models/efficientnet_b5_fine_tuned.pth"),
    "EfficientNet-B6": ("efficientnet_b6", "models/efficientnet_b6_fine_tuned.pth"),
    "SE-ResNet18": ("seresnet50", "models/seresnet50_fine_tuned.pth")
}

MODEL_CACHE = {}

def load_model(model_key):
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    name, path = MODEL_PATHS[model_key]
    if name == "vit":
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification(config)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    elif name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif name == "vgg16":
        from torchvision.models import vgg16
        model = vgg16(weights=None)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    elif name == "densenet121":
        from torchvision.models import densenet121
        model = densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    elif name.startswith("efficientnet"):
        from torchvision import models
        fn = getattr(models, name)
        model = fn(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif name == "seresnet50":
        model = timm.create_model("seresnet50", pretrained=False, num_classes=2)
    else:
        raise ValueError("Unsupported model")

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    MODEL_CACHE[model_key] = model
    return model

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": list(MODEL_PATHS.keys())
    })

@app.post("/predict_ui", response_class=HTMLResponse)
async def predict_ui(request: Request, file: UploadFile = File(...), model_key: str = Form(...), preprocess_type: str = Form(...)):
    torch.cuda.empty_cache()
    gc.collect()

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if preprocess_type == "auto":
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "No face detected! Please try a clearer photo.",
                "models": list(MODEL_PATHS.keys())
            })
        box = boxes[0]
        img = img.crop((box[0], box[1], box[2], box[3]))

    img_tensor = transform(img).unsqueeze(0).to(device)

    try:
        model = load_model(model_key)
        start = time.time()
        with torch.no_grad():
            output = model(img_tensor)
            if hasattr(output, "logits"): output = output.logits
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()
        end = time.time()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "CUDA out of memory. Try a lighter model or restart the app.",
                "models": list(MODEL_PATHS.keys())
            })
        raise e

    result = {
        "pred_class": "Fake" if pred_class == 1 else "Real",
        "confidence": f"{confidence*100:.2f}%",
        "time_ms": int((end - start) * 1000)
    }

    # Re-encode cropped image if applicable
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_data": encoded_img,
        "result": result,
        "model": model_key,
        "preprocess": preprocess_type,
        "models": list(MODEL_PATHS.keys())
    })
