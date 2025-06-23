import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from llama import Cardio_llama
from accelerate import Accelerator

def train_one_epoch(model: Cardio_llama,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    accelerator: Accelerator,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
   
    for data_iter_step, (examples, labels, example_mask, ecg, cxr, lab, modality) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # if (data_iter_step + 1) % 40000 == 0:
        #     print(f"Saving Model to ", args.output_dir)
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)
        #     print(f"Model Saved")

        # print(modality)
  
        ecg = ecg.to(device)
        cxr = cxr.to(device)
        lab = lab.to(device)
        
        with torch.cuda.amp.autocast():
            _, c_loss = model(examples, labels, ecgs=ecg, cxrs=cxr, labs=lab)
        
        loss = c_loss 
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        # m_loss_value = m_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(accelerator, loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        # if m_loss_value != 0:
        #     metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
 


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
