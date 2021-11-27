import torch
from skimage import io
import Network.Bank.bankset as bank
import Network.Bank.transforms as trans
import Network.parameters as par
import Network.Architecture.model as mod

#Varaibles - in tools variables are here to ease testing
MODEL_PATH = par.MODEL_LOAD_MODEL_PATH
MODEL_TYPE = par.MODEL_USED_MODEL_TYPE
MODEL_QUANTIZED = par.EVAL_LOAD_MODEL_IS_QUANTIZED
TEST_DEVICE = par.QUANT_DEVICE if par.EVAL_LOAD_MODEL_IS_QUANTIZED else par.TRAIN_DEVICE
Image_PATH = '../../../Data/testset/500/63.jpg'
Image_Label = 500


def main():

    used_model = mod.UsedModel(MODEL_TYPE, arg_load_path=MODEL_PATH, arg_load=True, arg_load_device=TEST_DEVICE)
    used_model.model.to(TEST_DEVICE)
    used_model.model.eval()

    transform_test = trans.TRANSFORM_BLANK

    image = io.imread(Image_PATH)
    image = transform_test(image)
    image = image[None,:,:,:]
    #image = image.to(test_device)

    with torch.no_grad():
        output = used_model.model(image)
        print(output)
        print("classes ",end="")
        print(bank.classes)
        pred_val, pred = output.topk(1, 1, True, True)
        pred_val = pred_val.numpy()
        print("Choosen nr "+ str(pred) + " With sertanty = " + str(pred_val))
        suggest = bank.anticlasses.get(pred.item())
        suggest_val = pred_val[0][0]
        correct = Image_Label
        correct_val = output.numpy()[0][0][bank.anticlasses.get(correct)][0]
        hit = pred.eq(bank.classes.get(str(Image_Label)))

        print("Image " + str(Image_PATH) + " recognised as " + str(suggest) + " zl, with " + str(suggest_val) + " certainty")


        if hit is True:
            print("This is Correct")
        else:
            print("This is Not Correct")

        print("This image should be " + "recognised as " + str(correct) + " zl, " + "( " + str(correct_val) + " )")



def multi():
    _,_,testloader = bank.loadData(arg_load_train=False, arg_load_val=False, arg_load_test=True)
    test_device = torch.device(par.TRAIN_DEVICE)

    # Empty GPU Cache before Testing starts
    if par.TRAIN_DEVICE == 'cuda:0': torch.cuda.empty_cache()

    used_model = mod.UsedModel(par.MODEL_USED_MODEL_TYPE, arg_load_path=par.MODEL_LOAD_MODEL_PATH, arg_load=True)
    used_model.to(TEST_DEVICE)
    used_model.model.eval()

    singleBatch = next(iter(testloader))
    input, label, name = singleBatch['image'], singleBatch['class'], singleBatch['name']

    with torch.no_grad():
        output = used_model.model(input)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.contiguous().view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        print(str(correct_k))

if __name__ == '__main__':
    main()