import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import time
import torch.onnx

##
# Based on https://nextjournal.com/gkoehler/pytorch-mnist
##

### High-level setup ###

TRAIN = False
LOAD_LATEST = False
WRITE_LOGS = False
SAVE_CKPTS = False
EXPORT_MODEL = False

MODEL_NAME = "MNIST-Classifier"
CLASS_COUNT = 10

EPOCHS = 3              # The number of Epochs (a complete pass through the training data)
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 100

LEARNING_RATE = 0.01
MOMENTUM = 0.5

LOG_INTERVAL = 100      # Interval of steps between each saved log
CKPT_INTERVAL = -1      # Interval of epochs between each saved checkpoint

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 01 Load data locally from the designated folders
trainData = datasets.ImageFolder('MNIST_Extracted/train', transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))
testData = datasets.ImageFolder('MNIST_Extracted/test', transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))

# 02 Prepare the training and test data, creating the tensors
trainSet = torch.utils.data.DataLoader(trainData, BATCH_SIZE_TRAIN, shuffle=True)
testSet = torch.utils.data.DataLoader(testData, BATCH_SIZE_TEST, shuffle=False)

# 03 Define the summary writer for saving Tensorboard compatible logs
if WRITE_LOGS:
    writer = SummaryWriter(log_dir=f"runs/{MODEL_NAME}-{time.time_ns()}")



# Define the Neural Network class, inherits from nn.Module
class Net(nn.Module):
    # The class constructor
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        self.fc1 = nn.Linear(320, 50)           # Result of the dropout, multiplied channels!
        self.fc2 = nn.Linear(50, CLASS_COUNT)   # Map the result back to the amount of classes
    
    # The forward method, the feeds data through the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # ouputs size([1, 20, 4, 4])
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)          # Return the highest probability
        
# 04 Construct the network and the optimizer
net = Net().to(DEVICE)
optimizer = optim.SGD(net.parameters(), lr= LEARNING_RATE, momentum= MOMENTUM)

##
# Functions
##

# 05 Define the training function
def Train(epoch):
    # 06 Set the model to training mode
    net.train()
    # 07 Iterate through the training data set
    for idx, (data, target) in enumerate(trainSet):
        # 08 Restart the optimizer
        optimizer.zero_grad()
        
        # 09 Send data to the same device (GPU/CPU) as Network
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        
        # 07 Compute the output
        output = net(data)
        
        # 08 Calculate loss and propagate back through the Network
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # 09 Log training information
        if idx % LOG_INTERVAL == 0:
            # 10 Print the loss on console
            print(f"Step {idx} of Epoch {epoch} Training Loss: {loss}")
            
            # 11 If set to, write the training logs
            if WRITE_LOGS:
                writer.add_scalar('Training Loss', loss, idx + (epoch * len(trainSet)))

# 12 Define the testing function
def TestOnData(epoch):
    # 13 Set the model to evaluation mode
    net.eval()

    # 14 Count the number of correct predictions 
    correct = 0
    total = 0

    # 15 Iterate through test set 
    with torch.no_grad():
        for data, target in testSet:
            # 16 Send data to the same device (GPU/CPU) as Network
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            # 17 Compute the output
            output = net(data)

            # 18 Check all predictions
            for idx, prediction in enumerate(output):
                # 19 Check if higher prediction matches the target
                if torch.argmax(prediction) == target[idx]:
                    correct += 1
                total +=1
        
        # 20 Calculate total accuracy over the test set and print
        acc = round(correct / total, 3)
        print(f"Epoch {epoch} accuracy: {acc}")
        
        # 21 If set to, write the training logs
        if WRITE_LOGS:
            writer.add_scalar('Accuracy', acc, epoch)


# 22 Test prediction on one image and visualize the result
def TestOnImage():
    # 23 Set the model to evaluation mode
    net.eval()

    # 24 Open the image to be tested from the project's folder on greyscale mode
    img = Image.open("data/number.png").convert("L")
    
    # 25 Transform image into a Tensor
    transform = transforms.ToTensor()
    img = transform(img)

    # 26 Run image on the network
    with torch.no_grad():
        # 27 Shape Tensor in the expected shape and send it to the same
        # device as the Network
        img = img.view(-1, 1, 28, 28).to(DEVICE)
        
        # 28 Compute the output
        output = net(img)

        # 29 Get the prediction's index
        prediction = torch.argmax(output)
        
        # 30 Use pyplot to display the image
        # Send the image to the CPU and shape it as a 28x28 grid
        plt.imshow(img.cpu().view(28,28))

        # 31 Get the prediction name from the index
        predictedClass = testData.classes[prediction.item()]

        # 32 Add prediction as image title
        plt.title("Prediction: " + str(predictedClass))
        
        # 33 Show image with pyplot
        plt.show()


# 34 Function for saving checkpoints
def SaveCheckpoint():
    torch.save({'epoch': epoch, 
                'model_state_dict': net.state_dict(), 
                'optmizer_state_dict': optimizer.state_dict()}, 
                f"checkpoints/{MODEL_NAME}-{time.time_ns()}.pt")
    print("Saved checkpoint")


# 43 Function for loading the latest saved checkpoint
def LoadLatest():
    checkpoint = ""

    # 44 Find the latest file
    list_of_files = glob.glob("checkpoints/*.pt")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Found existing model, loading {latest_file}")
    
    # 45 Load checkpoint into the Network and the Optimizer
    checkpoint = torch.load(latest_file)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optmizer_state_dict"])

##
# Execution
##

# 46 Load latest
if LOAD_LATEST:
    LoadLatest()
    TestOnImage()

# 35 Training loop
if TRAIN:
    # 36 Record the time the training session started
    start_time = time.time()

    # 37 Iterate through epochs
    for epoch in range(EPOCHS):
        Train(epoch)
        print('\n')
        TestOnData(epoch)
        print('\n')
        
        # 38 Save checkpoints
        if SAVE_CKPTS:
            # 39 Save only the last checkpoint
            if CKPT_INTERVAL == -1:
                if epoch == EPOCHS -1:
                    SaveCheckpoint()
            # 40 Save checkpoints between the set epoch interval
            elif epoch % CKPT_INTERVAL == 0:
                SaveCheckpoint()
    
    # 41 Close the summary writer
        if WRITE_LOGS:
          writer.close() # type: ignore
    
    # 42 Print the elapsed time and test on the saved image
    print(f"Finished in {time.time() - start_time} seconds")
    TestOnImage()

# 47 Export the current model
if EXPORT_MODEL:
    # 48 Create a random tensor shaped as the expected input data
    x = torch.randn(1, 1, 28, 28)
    
    # 49 Create the target folder if it does not exist yet
    if not os.path.exists('Exported'):
        os.makedirs('Exported')
    
    path = os.path.join('Exported', f"{MODEL_NAME}-{time.time_ns()}.onnx")
    f = open(path, 'w+')
    
    # 50 Export the model
    torch.onnx.export(net, x.to(DEVICE), path, export_params = True, opset_version=10)