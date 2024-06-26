# %%
# 导出torchscript模型
import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
dummy_input = torch.rand(1, 3, 224, 224)

trace_model = torch.jit.trace(model, dummy_input)
script_model = torch.jit.script(model)
("resnet18.pt")
print("torchscript model exported!")
output = trace_model(dummy_input)
print(output.shape)

# 保存模型
trace_model.save("./logs/jit_script_model/trace_resnet50.pt")
script_model.save("./logs/jit_script_model/script_resnet50.pt")
# %%
