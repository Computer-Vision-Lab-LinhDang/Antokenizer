from train_multimodal_ddp import MultiModalLitModule
from mavt.mavt_cls import MAVTClassifier
import torch

checkpoint_path = "/home/sagemaker-user/ws/Antoken/checkpoints/multimodal/multimodal-1-step=50000.ckpt"
base_model = MultiModalLitModule.load_from_checkpoint(
    checkpoint_path=checkpoint_path, device="cuda:0"
)
classifier = MAVTClassifier(num_classes=10, classifier_name="fastformer")
classifier = classifier.to("cuda:0")

for param in base_model.parameters():
    param.requires_grad = False
base_model.eval()
classifier.patchify = base_model.model.patchify
classifier.encoder = base_model.model.encoder
classifier.latent = base_model.model.latent
classifier.decoder = base_model.model.decoder 

image = torch.randn(1, 3, 224, 224).to("cuda:0")
classifier.eval()
with torch.no_grad():
    outputs = classifier(image)
print(outputs[0].reconstruction.shape)
print(outputs[0].aux)
for out in outputs[0].aux:
    print(out.shape)
print(outputs[1].z.shape)
print(outputs[1].z_understand.shape)
print(outputs[1].mu.shape)
print(outputs[1].log_var.shape)
print(outputs[2].shape)

video = torch.randn(1, 3, 16, 224, 224).to("cuda:0")
outputs = classifier(video)
print(outputs[0].reconstruction.shape)
print(outputs[0].aux)
for out in outputs[0].aux:
    print(out.shape)
print(outputs[1].z.shape)
print(outputs[1].z_understand.shape)
print(outputs[1].mu.shape)
print(outputs[1].log_var.shape)