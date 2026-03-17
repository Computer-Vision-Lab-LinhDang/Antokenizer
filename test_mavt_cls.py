from mavt.tokenizer import MAVTokenizer
from mavt.config import MAVTConfig
from mavt import MAVTClassifier
import torch



tokenizer = MAVTClassifier(num_classes=10, classifier_name="fastformer")
image = torch.randn(1, 3, 224, 224)
tokenizer.eval()
with torch.no_grad():
    outputs = tokenizer(image)
print(outputs[0].reconstruction.shape)
print(outputs[0].aux)
for out in outputs[0].aux:
    print(out.shape)
print(outputs[1].z.shape)
print(outputs[1].z_understand.shape)
print(outputs[1].mu.shape)
print(outputs[1].log_var.shape)
print(outputs[2].shape)

# video = torch.randn(1, 3, 16, 224, 224)
# outputs = tokenizer(video)
# print(outputs[0].reconstruction.shape)
# print(outputs[0].aux)
# for out in outputs[0].aux:
#     print(out.shape)
# print(outputs[1].z.shape)
# print(outputs[1].z_understand.shape)
# print(outputs[1].mu.shape)
# print(outputs[1].log_var.shape)