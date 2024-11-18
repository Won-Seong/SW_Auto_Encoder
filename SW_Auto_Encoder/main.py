import torch
from auto_encoder.auto_encoder import AutoEncoder
import os

from helper.data_generator import DataGenerator
from helper.painter import Painter
from helper.trainer import Trainer

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    dg = DataGenerator()
    #dl = dg.fashion_mnist(batch_size = 128)
    dl = dg.noise(size = (1000, 3, 512, 512))
    pt = Painter()
    
    
    #a = next(iter(dl))[0][0:2]
    #pt.show_or_save_images(a)
    
    ae = AutoEncoder(config_path = 'configs/noise512_config.yaml', device = device)
    #ae.load(file_name = 'ae_epoch2')
    #d = ae.reconstruct(a)
    #pt.show_or_save_images(d)
    
    tr = Trainer(ae, loss_fn = ae.loss)
    tr.train(dl, epochs = 10, file_name = 'ae')
    
    



