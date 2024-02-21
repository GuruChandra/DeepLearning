import torch
from pathlib import Path
import matplotlib.pyplot as plt 

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj = model.state_dict(),
               f=model_save_path)


def plot_loss_curves(data):
    epochs = range(len(data["train_loss"]))
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(epochs, data["train_loss"],  label='train_loss')
    plt.plot(epochs, data["test_loss"],  label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, data["train_acc"], label='train_acc')
    plt.plot(epochs, data["test_acc"],  label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.show()

    print(epochs)
    