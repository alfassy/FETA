import os
import clip
import torch
from torchvision.datasets import CocoCaptions, ImageNet
from torch import nn, einsum
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100
from sklearn.metrics import average_precision_score


def zero_shot_test_CIFAR(model=None, test_data_loader=None, clip_preprocess=None, batch_size=16, num_workers=8):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, clip_preprocess = clip.load('ViT-B/32', device)
    else:
        _, clip_preprocess = clip.load('ViT-B/32', device)
    val_acc = []
    model.eval()

    # Download the dataset
    test_dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=clip_preprocess)
    # test_data_loader = ImageNet(root='/dccstor/alfassy/data/imagenet', download=False, train=False)
    # test_data_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder('/dccstor/leonidka1/data/imagenet/ILSVRC/Data/CLS-LOC/val/', transform=clip_preprocess),
    #     batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for test_data in test_data_loader:
        # Prepare the inputs
        image_input, gt_labels = test_data
        text_inputs = torch.stack([clip.tokenize(f'a photo of a {c}') for c in test_data_loader.dataset.classes]).squeeze()
        image_input, text_inputs, gt_labels = image_input.cuda(), text_inputs.cuda(), gt_labels.cuda()

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        indices = similarity.argmax(dim=1)
        # map_acc = average_precision_score(values, gt_labels)
        # val_acc.append(calc_acc(indices.cpu().numpy(), gt_labels))
        val_acc.append((gt_labels == indices).sum() / indices.shape[0])
        # Print the result
    return sum(val_acc) / len(val_acc)

def zero_shot_test_imagenet(model=None, data_loader=None, clip_preprocess=None, batch_size=16, num_workers=8):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, clip_preprocess = clip.load('ViT-B/32', device)
    else:
        _, preprocess = clip.load('ViT-B/32', device)
    val_acc = []
    model.eval()

    # Download the dataset
    # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    test_data_loader = ImageNet(root='/dccstor/alfassy/data/imagenet', download=False, train=False)
    # test_data_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder('/dccstor/leonidka1/data/imagenet/ILSVRC/Data/CLS-LOC/val/', transform=clip_preprocess),
    #     batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for test_data in test_data_loader:
        # Prepare the inputs
        image_input, gt_labels = test_data
        text_inputs = torch.stack([clip.tokenize(f'a photo of a {c}') for c in data_loader.dataset.coco_class_list]).squeeze()
        image_input, text_inputs = image_input.cuda(), text_inputs.cuda()

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        similarity = (100.0 * image_features @ text_features.T).sigmoid()
        values, indices = similarity
        map_acc = average_precision_score(values, gt_labels)
        val_acc.append(calc_acc(indices.cpu().numpy(), gt_labels_indices))
        # Print the result
    return sum(val_acc) / len(val_acc)


def zero_shot_test_COCO(model=None, data_loader=None, clip_preprocess=None, batch_size=16, num_workers=8):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, clip_preprocess = clip.load('ViT-B/32', device)
    else:
        _, preprocess = clip.load('ViT-B/32', device)
    val_acc = []
    model.eval()

    for test_data in data_loader:
        # Prepare the inputs
        image_input, gt_labels = test_data
        text_inputs = torch.stack([clip.tokenize(f'a photo of a {c}') for c in data_loader.dataset.coco_class_list]).squeeze()
        image_input, text_inputs = image_input.cuda(), text_inputs.cuda()

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        similarity = (100.0 * image_features @ text_features.T).sigmoid()
        values, indices = similarity
        map_acc = average_precision_score(values, gt_labels)
        val_acc.append(calc_acc(indices.cpu().numpy(), gt_labels_indices))
        # Print the result
    return sum(val_acc) / len(val_acc)


def clip_batch_acc(scores, gt_labels):
    pred_labels = torch.argmax(scores, dim=0)
    acc = len(torch.where(pred_labels==gt_labels)[0])/len(gt_labels)
    return acc


class CLIP_dalle(nn.Module):
    def __init__(
            self,
            *,
            dim_text = 512,
            dim_image = 512,
            dim_latent = 512,
            num_text_tokens = 10000,
            text_enc_depth = 6,
            text_seq_len = 256,
            text_heads = 8,
            num_visual_tokens = 512,
            visual_enc_depth = 6,
            visual_heads = 8,
            visual_image_size = 256,
            visual_patch_size = 32,
            channels = 3
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer(causal = False, seq_len = text_seq_len, dim = dim_text, depth = text_enc_depth, heads = text_heads)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        assert visual_image_size % visual_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (visual_image_size // visual_patch_size) ** 2
        patch_dim = channels * visual_patch_size ** 2

        self.visual_patch_size = visual_patch_size
        self.to_visual_embedding = nn.Linear(patch_dim, dim_image)
        self.visual_pos_emb = nn.Embedding(num_patches, dim_image)
        self.visual_transformer = Transformer(causal = False, seq_len = num_patches, dim = dim_image, depth = visual_enc_depth, heads = visual_heads)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        self.temperature = nn.Parameter(torch.tensor(1.))

    def forward(
            self,
            text,
            image,
            text_mask = None,
            return_loss = False
    ):
        b, device, p = text.shape[0], text.device, self.visual_patch_size

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        image_patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        image_emb = self.to_visual_embedding(image_patches)
        image_emb += self.visual_pos_emb(torch.arange(image_emb.shape[1], device = device))

        enc_text = self.text_transformer(text_emb, mask = text_mask)
        enc_image = self.visual_transformer(image_emb)

        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim = 1)
        else:
            text_latents = enc_text.mean(dim = 1)

        image_latents = enc_image.mean(dim = 1)

        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = torch.arange(b, device = device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

def average_precision_compute_fn(y_preds, y_targets, mask, activation=None):

    y_true = y_targets.numpy()
    if activation is not None:
        y_preds = activation(y_preds)
    y_pred = y_preds.numpy()

    if mask is not None:
        y_true = y_true[:, mask]
        y_pred = y_pred[:, mask]

    return average_precision_score(y_true, y_pred)


def calc_acc(pred_labels, gt_labels):
    correct = 0.0
    total = float(len(gt_labels))
    for label in gt_labels:
        if label in pred_labels:
            correct += 1.0
    return correct / total



if __name__ == '__main__':
    import pydevd_pycharm
    pydevd_pycharm.settrace('9.145.9.103', port=55551, stdoutToServer=True, stderrToServer=True, suspend=False)
    acc = zero_shot_test_CIFAR()
    print(f'acc: {acc}')
