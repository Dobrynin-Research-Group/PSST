import numpy as np
import torch

import mike_torch_lib as mike

class NeuralNet(torch.nn.Module):   
    """The classic, fully connected neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self, shape):
        """Input:
                np.array of size 32x32 of type np.float32
        
        Three fully connected layers. 
        Shape of data progresses as follows:

                Input:          (32, 32)
                Flatten:        (1024,) [ = 32*32]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
        """
        super(NeuralNet, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(shape[0]*shape[1], 128), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.conv_stack(x)

class ConvNeuralNet(torch.nn.Module):   
    """The convolutional neural network.
    TODO: Make hyperparameters accessible and tune.
    """
    def __init__(self, shape):
        """Input:
                np.array of size 32x32 of type np.float32
                Two convolutional layers, three fully connected layers. 
                Shape of data progresses as follows:

                Input:          (32, 32)
                Unflatten:      ( 1, 32, 32)
                Conv2d:         ( 6, 30, 30)
                Pool:           ( 6, 15, 15)
                Conv2d:         (16, 13, 13)
                Pool:           (16,  6,  6)
                Flatten:        (576,) [ = 16*6*6]
                FCL:            (64,)
                FCL:            (64,)
                FCL:            (3,)
        """
        super(ConvNeuralNet, self).__init__()

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Unflatten(1, (1, 32)),
            torch.nn.Conv2d(1, 6, 3), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(6, 16, 3), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2),
            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(16*6*6, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.conv_stack(x)

def get_data(generator, processor, batch_size):
    y = np.random.random((batch_size, 3)).astype(np.float32)
    Bg, Bth, Pe = processor.unnormalize_params(*(y.T))
    eta_sp = generator.generate(Bg, Bth, Pe)
    eta_sp = processor.add_noise(eta_sp)
    eta_sp = processor.normalize_visc(eta_sp)
    X = processor.cap(eta_sp).astype(np.float32)
    return X, y

def train(generator, processor, 
        model, loss_fn, optimizer, device,
        num_samples, batch_size):
    model.train()
    num_batches = num_samples // batch_size
    print_every = 5
    for b in range(num_batches):
        X, y = get_data(generator, processor, batch_size)
        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (b + 1) % print_every == 0:
            loss, current = loss.item(), (b + print_every) * batch_size
            print(f'[{current:>7d}/{num_samples:>7d}]')
            mean_error = torch.mean(torch.abs(y - pred) / y, 0)
            print(f'\tmean_error = {mean_error[0]} {mean_error[1]} {mean_error[2]}')
            print(f'\t{loss = :>7f}')

def test(generator, processor, 
        model, loss_fn, device,
        num_samples, batch_size):
    model.eval()
    avg_loss = 0
    avg_error = 0
    num_batches = num_samples // batch_size
    with torch.no_grad():
        for b in range(num_batches):
            X, y = get_data(generator, processor, batch_size)
            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += loss.item()
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)
    
    avg_loss /= num_batches
    avg_error /= num_batches

    print(f'Accuracy:\n\t{avg_loss = :>5f}\n\taverage errors ='
        f' {avg_error[0]:>5f} {avg_error[1]:>5f} {avg_error[2]:>5f}'
    )

def main():
    
    batch_size = 1000
    train_size = 500000
    test_size = 100000

    generator = mike.SurfaceGenerator('surface_bins.json')
    processor = mike.Processor(
        data_file='surface_bins.json',
        param_file='Bg_Bth_Pe_range.json'
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'{device = }')

    shape = generator.generate(
        np.array([0.8]),
        np.array([0.8]),
        np.array([8])
    )[0].shape

    model = NeuralNet(shape).to(device)
    print('Loaded model.')

    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(2):
        print(f'\n*** Epoch {i+1} ***')
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=0.1, 
            momentum=0.9/(i+1)
        )

        print('Training')
        train(generator, processor, 
            model, loss_fn, optimizer, device,
            train_size, batch_size
        )

        print('Testing')
        test(generator, processor,
            model, loss_fn, device,
            test_size, batch_size
        )

if __name__ == '__main__':
    main()