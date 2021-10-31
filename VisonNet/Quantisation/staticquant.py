import torch
import torch.quantization as quant
import torch.backends.quantized as backquant
import torchvision.models.quantization as quantmodels
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import enum

import Network.parameters as par
import Network.bankset as bank
import Network.model as mod
import Network.evaluate as eva

BACKEND_ENGINE = ''  #quantization engine
DO_EVALUATE = False
quantDevice = None

def quantMain():

    # Choose quantization engine
    if 'qnnpack' in backquant.supported_engines:
        # This Engine Works ONLY on Linux
        # We will use it
        print("Using qnnpack backend engine")
        BACKEND_ENGINE = 'qnnpack'
    elif 'fbgemm' in backquant.supported_engines:
        # This Engine works on Windows (and Linux?)
        # We won't be using it
        BACKEND_ENGINE = 'fbgemm'
        print("FBGEMM Backend Engine is not supported - are you trying this on windows?")
        exit(-2)
    else:
        BACKEND_ENGINE = 'none'
        print("No Proper Backend Engine found")
        exit(-3)


    # Choose quantization device (cpu/gpu)
    # Static Quantisation works only on cpu
    quantDevice = par.QUANT_DEVICE


    # Load Data
    #TODO: transforms
    transform_for_quant = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(224),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701],
                                                              std=[0.24467267, 0.23742135, 0.24701703]), ])

    dataset_loader, _, _ = bank.loadData(arg_load_train=True, arg_load_val=False, arg_load_test=False,
                                         arg_trans_train=transform_for_quant, quantisation_mode=True)

    #Load Our Model
    quant_model = mod.UsedModel(par.USED_MODEL_TYPE, arg_load=True, arg_load_path=par.QUANT_MODEL_PATH, arg_load_device=par.QUANT_DEVICE,arg_load_raw=True)
    quant_model.optimizer = torch.optim.Adam(quant_model.model.parameters(), lr=par.INITIAl_LEARNING_RATE) ##only if raw load
    quant_model.model.to(par.QUANT_DEVICE)
    print('Loaded trained model')

    quant_model.model.eval()
    quant_model.addQuantStubs() #needed???? for old 1.6  way

    quant_model.fuzeModel()
    #quant_model.model.fuse_model()

    # Evaluate Our Model
    if DO_EVALUATE:
        print("Started Evaluation")
        quant_model.model.eval()
        top1, top5 = eva.evaluate(quant_model, dataset_loader, par.QUANT_DEVICE)
        print('Evaluation accuracy on all test images, %2.2f' % (top1.avg))


   # propagation_list = quant.get_default_qconfig_propagation_list()
   # propagation_list.remove(torch.nn.modules.linear.Linear)
   # q_config_dict = dict()
   # for e in propagation_list:
   #     q_config_dict[e] = quant.get_default_qconfig(BACKEND_ENGINE)
   # quant.propagate_qconfig_(quant_model.model, q_config_dict)

    white_list = torch.quantization.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
    white_list.remove(torch.nn.modules.linear.Linear)
    qconfig_dict = dict()
    quant_model.model.eval()
    for e in white_list:
        qconfig_dict[e] = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.propagate_qconfig_(quant_model.model, qconfig_dict)



    #quant_model.model.qconfig = quant.default_qconfig
    quant.prepare(quant_model.model, inplace=True)

    #Calibrate
    print("\nStarting Quantizising Imputs")
    quant_model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset_loader, 0):
            if (i+1) % 2 == 0: break
            if i % 1000 == 0: print("Progress = ", i)
            inputs, labels = data['image'], data['class']
            quant_model.model(inputs)
    print("Imputs Quantized")

    #Convert to quantized model
    torch.quantization.convert(quant_model.model, inplace=True)
    print("Model Quantized")

    # Evaluate Our Model
    if DO_EVALUATE:
        print("Started Evaluation")
        quant_model.model.eval()
        top1, top5 = eva.evaluate(quant_model, dataset_loader, par.QUANT_DEVICE) #todo? QuantizedCPU
        print('Evaluation accuracy on all test images, %2.2f' % (top1.avg))


    # save for mobile
    print("Started Saving Model")
    for i, data in enumerate(dataset_loader):
        inputs, labels = data['image'], data['class']
        traced_script_module = torch.jit.trace(quant_model.model, inputs)
        traced_script_module.save("rn18quantized.pt")
        break

    print('Model Saved Successfully')
    return


    #quant_model.addQuantStubs()

    print('\nQuantization Started')

    torch.backends.quantized.engine = BACKEND_ENGINE

    white_list = torch.quantization.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
    # white_list = torch.quantization.get_default_qconfig_propagation_list()
    # xx = torch.quantization.get_default_qat_module_mappings()
    white_list.remove(torch.nn.modules.linear.Linear)
    qconfig_dict = dict()
    quant_model.model.eval()
    for e in white_list:
        qconfig_dict[e] = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.propagate_qconfig_(quant_model.model, qconfig_dict)
    quant_model.model.eval()

    quant_model.model.eval()


    # v = model.model(trainloader.dataset.images[0]['class'])





    if DO_EVALUATE:
        best_acc = 0
        num_train_batches = 4
        quant_model.model.to(quantDevice)
        quant_model.model.eval()
        top1, top5 = eva.evaluate(quant_model.model, dataset_loader, quantDevice)
        print('Evaluation accuracy on all test images, %2.2f' % top1.avg)




if __name__ == '__main__':
    quantMain()
