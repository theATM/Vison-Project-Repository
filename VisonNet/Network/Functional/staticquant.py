import torch
import torch.quantization as quant
import torch.backends.quantized as backquant
from torchvision import transforms

import Network.parameters as par
import Network.Bank.bankset as bank
import Network.Bank.transforms as trans
import Network.Architecture.model as mod
import Network.Functional.evaluate as eva

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
    transform_for_quant = trans.TRANSFORM_QUANTIZE

    dataset_loader, _, _ = bank.loadData(arg_load_train=True, arg_load_val=False, arg_load_test=False,
                                         arg_trans_train=transform_for_quant, quantisation_mode=True)

    #Load Our Model
    quant_model = mod.UsedModel(par.USED_MODEL_TYPE, arg_load=True, arg_load_path=par.QUANT_MODEL_PATH, arg_load_device=par.QUANT_DEVICE,arg_load_raw=par.LOAD_RAW_MODEL)
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


    propagation_list = quant.get_default_qconfig_propagation_list()
    propagation_list.remove(torch.nn.modules.linear.Linear)
    q_config_dict = dict()
    for e in propagation_list:
        q_config_dict[e] = quant.get_default_qconfig(BACKEND_ENGINE)
    quant.propagate_qconfig_(quant_model.model, q_config_dict)

    #white_list = torch.quantization.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
    #white_list.remove(torch.nn.modules.linear.Linear)
    #qconfig_dict = dict()
    #quant_model.model.eval()
    #for e in white_list:
    #    qconfig_dict[e] = torch.quantization.get_default_qconfig('qnnpack')
    #torch.quantization.propagate_qconfig_(quant_model.model, qconfig_dict)



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

    print("Started Evaluation")
    quant_model.model.eval()
    top1, top5 = eva.evaluate(quant_model, dataset_loader, par.QUANT_DEVICE) #todo? QuantizedCPU
    print('Evaluation accuracy on all test images, %2.2f' % (top1.avg))


    # save for mobile
    print("Started Saving Model")
    for i, data in enumerate(dataset_loader):
        inputs, labels = data['image'], data['class']
        traced_script_module = torch.jit.trace(quant_model.model, inputs)
        traced_script_module.save(par.QUANT_SAVE_MODEL_PATH)
        break

    print('Model Saved Successfully')




if __name__ == '__main__':
    quantMain()
