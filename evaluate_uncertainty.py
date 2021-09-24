import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging

from utils import strings
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from models import *
from datasets import *


def remove_prefix(s, prefix):
    return s[len(prefix):] if s.startswith(prefix) else s

def load_state_dict_into_module(state_dict, module):
    own_state = module.state_dict()
    for name, param in state_dict.items():
        name = remove_prefix(name, '_model.')
        name = remove_prefix(name, 'module.')
        if name in own_state:
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].resize_as_(param)
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
            

def restore(filename, model, include_params="*", exclude_params=()):
    # -----------------------------------------------------------------------------------------
    # Make sure file exists
    # -----------------------------------------------------------------------------------------
    if not os.path.isfile(filename):
        logging.info("Could not find checkpoint file '%s'!" % filename)
        quit()

    # -----------------------------------------------------------------------------------------
    # Load checkpoint from file including the state_dict
    # -----------------------------------------------------------------------------------------
    checkpoint_with_state = torch.load(filename, map_location="cpu")

    # -----------------------------------------------------------------------------------------
    # Load filtered state dictionary
    # -----------------------------------------------------------------------------------------
    state_dict = checkpoint_with_state['state_dict']
    restore_keys = strings.filter_list_of_strings(
        state_dict.keys(),
        include=include_params,
        exclude=exclude_params)

    state_dict = {key: value for key, value in state_dict.items() if key in restore_keys}

    # if parameter lists are given, don't be strict with loading from checkpoints
    strict = True
    if include_params != "*" or len(exclude_params) != 0:
        strict = False

    load_state_dict_into_module(state_dict, model)


def calibration(prob_correct, bins, binning):
    bin_acc = []
    bin_pred = []
    bin_examples = []
    bin_start_index = 0
    for i in range(bins):
        acc = 0
        pred = 0
        examples = 0
        if binning == 'equidistant':
            exit_criterion = bin_start_index + examples < len(prob_correct) and not prob_correct[bin_start_index + examples, 0] > (i+1) / bins
        if binning == 'equal_bin_size':
            exit_criterion = bin_start_index + examples < len(prob_correct) and examples < len(prob_correct) / bins
        while exit_criterion:
            pred += prob_correct[bin_start_index + examples, 0]
            acc += prob_correct[bin_start_index + examples, 1]
            examples += 1
            if binning == 'equidistant':
                exit_criterion = bin_start_index + examples < len(prob_correct) and not prob_correct[bin_start_index + examples, 0] > (i+1) / bins
            if binning == 'equal_bin_size':
                exit_criterion = bin_start_index + examples < len(prob_correct) and examples < len(prob_correct) / bins
        if examples > 0:
            bin_pred.append(pred /  examples)
            bin_acc.append(acc / examples)
            bin_examples.append(examples)
        bin_start_index += examples

    return bin_acc, bin_pred, bin_examples


def rejection_accuracy(entropy_correct):
    accs = []
    auc = 0
    examples = len(entropy_correct)
    incorrect = len(entropy_correct) - sum(entropy_correct[:,1])
    for i in range(len(entropy_correct)-1):
        examples -= 1
        incorrect -= (1 - entropy_correct[i,1])
        accs.append(incorrect / examples)
        auc += incorrect / examples
    auc /= len(entropy_correct)
    return accs, auc



def main():
    device = torch.device(ARGS['device'])
    model_name  = ARGS['net'] + ARGS['mode']
    if ARGS['num_classes'] > 0:
        ARGS['model_parmas'] = {'num_classes': ARGS['num_classes']}
    else:
        ARGS['model_parmas'] =  {'args': None}
    Model = globals()[model_name]
    model = Model(**ARGS['model_parmas'])
    restore(ARGS['filename'], model)
    model.eval()
    model.to(device)

    Dataset = globals()[ARGS['dataset']]
    dataset = Dataset(None, ARGS['dataset_path'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

    prob_correct = []
    entropy_correct = []
    nllh = 0.0
    for i, data_dict in enumerate(dataloader):
        with torch.no_grad():
            input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
            target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
            tensor_keys = input_keys + target_keys
            for key, value in data_dict.items():
                if key in tensor_keys:
                    data_dict[key] = value.to(device)
            output_dict = model(data_dict)
            target = data_dict['target1']

            if ARGS['mode'] == '':
                prediction = output_dict['output1']
                p = F.softmax(prediction, dim=1)
            elif 'VI' in ARGS['mode']:
                samples = 64
                prediction_mean = output_dict['prediction_mean'].unsqueeze(dim=2).expand(-1, -1, samples)
                prediction_variance = output_dict['prediction_variance'].unsqueeze(dim=2).expand(-1, -1, samples)
                normal_dist = torch.distributions.normal.Normal(torch.zeros_like(prediction_mean), torch.ones_like(prediction_mean))
                normals = normal_dist.sample()
                prediction = prediction_mean + torch.sqrt(prediction_variance) * normals
                p = F.softmax(prediction, dim=1).mean(dim=2)

            nllh -= torch.log(p[range(p.shape[0]), target]).sum().item()
            predicted = torch.argmax(p, dim=1)
            for j in range(len(target)):
                prob = p[j,:].max().item()
                predicted_correctly = float(predicted[j]==target[j])
                prob_correct.append([prob, predicted_correctly])
                pred_dist = Categorical(probs=p[j,:])
                entropy = pred_dist.entropy().cpu()
                entropy_correct.append([entropy, predicted_correctly])

    prob_correct = np.array(sorted(prob_correct, key=lambda x: x[0]))
    entropy_correct = np.array(sorted(entropy_correct, key=lambda x: -x[0]))
    print('Accuracy:', prob_correct[:,1].sum() / len(prob_correct[:,1]))
    print('NLLH:', nllh / len(prob_correct[:,1]))

    bin_acc, bin_pred, bin_examples = calibration(prob_correct, bins=20, binning='equidistant')
    abs_diff = [abs(bin_acc[i] - bin_pred[i]) * bin_examples[i] for i in range(len(bin_acc))]
    ece = sum(abs_diff) / len(prob_correct)
    print('ECE:', ece)

    accs, auc = rejection_accuracy(entropy_correct)
    print('AUMRC:', auc)


    k = len(accs) // 20
    print('MR10%', accs[2*k])
    print('MR25%', accs[5*k])
    print('MR50%', accs[10*k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='')
    parser.add_argument('--net', default='')
    parser.add_argument('--mode', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_classes', default=10, type=int)
    ARGS = vars(parser.parse_args())
    main()
