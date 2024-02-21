import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
from torchinfo import summary

NUM_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001


train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "cpu" #cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
  transforms.Resize((128,128)),
  transforms.TrivialAugmentWide(num_magnitude_bins=31),
  transforms.ToTensor()
])

train_transform = transforms.Compose([
  transforms.Resize((128,128)),
  #transforms.TrivialAugmentWide(num_magnitude_bins=31),
  transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir= train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

    



model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)
                              ).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr = LEARNING_RATE)


summary(model, input_size=[1, 3, 128, 128])



results = engine.train(model = model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

utils.plot_loss_curves(results)

utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_scripts_mode_tinyvgg_model.pth")


