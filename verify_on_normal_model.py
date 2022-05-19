import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pretrainedmodels
from utils.CustomDataset import CustomDataset2


def verify(source_model, batchsize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_models = {}
    model_inc_v3 = pretrainedmodels.__dict__["inceptionv3"](num_classes=1000, pretrained='imagenet')
    model_inc_v4 = pretrainedmodels.__dict__["inceptionv4"](num_classes=1000, pretrained='imagenet')
    model_incres_v2 = pretrainedmodels.__dict__["inceptionresnetv2"](num_classes=1000, pretrained='imagenet')
    model_res_101 = pretrainedmodels.__dict__["resnet152"](num_classes=1000, pretrained='imagenet')
    model_res_152 = pretrainedmodels.__dict__["resnet101"](num_classes=1000, pretrained='imagenet')
    model_vgg_19 = pretrainedmodels.__dict__["vgg19"](num_classes=1000, pretrained='imagenet')
    model_inc_v3.to(device)
    model_inc_v4.to(device)
    model_incres_v2.to(device)
    model_res_152.to(device)
    model_res_101.to(device)
    model_vgg_19.to(device)
    target_models["inc_v3"] = model_inc_v3
    target_models["inc_v4"] = model_inc_v4
    target_models["vgg_19"] = model_vgg_19
    target_models["res_101"] = model_res_101
    target_models["res_152"] = model_res_152
    target_models["incres_v2"] = model_incres_v2

    for target_model_name in target_models:

        target_model = target_models[target_model_name]
        if target_model_name == "inc_v3" or target_model_name == "inc_v4" or target_model_name == "incres_v2":
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        adv_dir = "storage/adv"
        sm_label_dir = "storage/label/" + source_model + "_right_label.txt"
        tm_label_dir = "storage/label/" + target_model_name + "_right_label.txt"
        dataset = CustomDataset2(adv_dir, sm_label_dir, tm_label_dir, transform)
        dataloader = DataLoader(dataset, batchsize)

        count = 0
        count_sceucess = 0

        for img, tag, name in dataloader:
            count = count + len(tag)
            img = img.to(device) * 2 - 1.
            tag = tag.to(device)
            target_model.eval()

            with torch.no_grad():
                pred = target_model(img).argmax(1)
            count_sceucess = count_sceucess + sum(tag != pred).item()
            print("\r", "[Source model:[{}] || Target model:[{}] || No.{}, Success:{}, SR:{}"
                  .format(source_model, target_model_name, count, count_sceucess, round(count_sceucess / count, 4)),
                  end="",
                  flush=True)
        print("\n")


if __name__ == "__main__":
    source_model = "inc_v3"
    batchsize = 50
    verify(source_model, batchsize)
