import numpy as np
import torch

def eval_cal(output,mask):
    print(output.shape)
    output=torch.Tensor(output)
    output_flat=output.view(-1)
    mask_flat=mask.view(-1)
    output_flat=output_flat.numpy()
    mask_flat=mask_flat.numpy()
    output_flat=np.where(output_flat==255,1,0)
    # mask_flat=np.where(mask_flat==255,1,0)
    P_output=output_flat.sum()
    P_mask=mask_flat.sum()
    print("P_output:{}".format(P_output))
    print("P_mask:{}".format(P_mask))
    # TP=0
    # for i in range(output_flat.size):
    #     print(output_flat[i])
    #     print(mask_flat[i])
    #     if output_flat[i]==1 and mask_flat[i]==1:
    #
    #         TP+=1
    TP=(output_flat * mask_flat).sum()
    print(TP)
    print('TP:{}'.format(TP))
    FN=P_mask-TP
    print('FN:{}'.format(FN))
    FP=P_output-TP
    print("FP:{}".format(FP))
    TN=output_flat.size-(TP+FP+FN)

    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*(precision*recall/(precision+recall))

    return precision,recall,F1
    #FFFGGFG