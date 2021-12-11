"""

Run experiments

"""


import argparse
import json
import os
import pdb
import utils
import sklearn.utils
import torch
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from datasets import QaDataset
from evaluate_squad2 import evaluate
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import logging
logger = logging.getLogger(__name__)
MAXPROB_SOURCE_PROBS = {}
MAXPROB_TARGET_PROBS = {}
CAL_SOURCE_PROBS = {}
CAL_TARGET_PROBS = {}
TTDO_SOURCE_PROBS = {}
TTDO_TARGET_PROBS = {}
OVERCONF_LIST = []
UNDERCONF_LIST = []

def initialize_args():
    parser = argparse.ArgumentParser(description='Extrapolation expts')
    parser.add_argument('model_dir', type=str, default=None,
                        help='Path to model params and cache directory')
    parser.add_argument('task', type=str, default=None,
                        choices=['nli', 'qa'], help='Task type')
    parser.add_argument('mode', type=str, default=None,
                        choices=['maxprob', 'extrapolate', 'wilcoxon',
                                 'minimum', 'maxprob_squad_only', 'ttdo',
                                 'squad2', 'outlier_detection', 'nn_train', 'nn_eval'],
                        help='Mode of execution')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to store probabilities, AUC')
    parser.add_argument('--ttdo_type', type=str, default='mean',
                        choices=['mean', 'neg_var'], help='TTDO mean or -var?')
    parser.add_argument('--ttdo_calibrator', action='store_true',
                        help='Use TTDO features in calibrator')
    parser.add_argument('--target_prefix', type=str, default=None,
                        help='Dataset prefix of target data.')
    parser.add_argument('--expose_prefix', type=str, default=None,
                        help='Dataset prefix of exposure data')
    parser.add_argument('--oracle', action='store_true',
                        help='Classifier has knowledge of domain.')
    parser.add_argument('--per_domain', action='store_true',
                        help='Print per-domain accuracy/coverage')
    parser.add_argument('--classifier', type=str, default='random_forest',
                        choices = ['logreg', 'svm', 'random_forest',
                                    'xgboost', 'knn'],
                        help='Classifier type for calibrator')
    parser.add_argument('--reg', type=float, default=6.0,
                        help='Regularization value for classifier')
    parser.add_argument('--reg2', type=int, default=175,
                        help='num estimators')
    parser.add_argument('--fraction_id', type=float, default=None,
                        help='fraction of D_cal/test that is in-domain')
    parser.add_argument('--ablate', type=str, default=None,
                            choices=['maxprob', 'other_prob', 'all_prob',
                                        'context_len', 'pred_len'],
                            help='Which feature(s) to ablate')
    parser.add_argument('--error_analysis', action='store_true',
                        help='Perform error analysis')
    parser.add_argument('--strict_eval', action='store_true',
                        help='Only consider one ans option correct')


    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=15, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    args = parser.parse_args()

    return args


def test_time_dropout(args, sourceDataset, targetDataset):
    # Get mean or neg-var (based on args.ttdo_type)
    # probabilities of main model prediction across 
    # 30 TTDO runs of the model.
    source_probs = utils.get_average_probs(args, sourceDataset, 'squad1.1')
    target_probs = utils.get_average_probs(args, targetDataset, args.target_prefix)
    probs_list = np.asarray([source_probs[guid] \
                                for guid in sourceDataset.dev_guid_list] + \
                                [target_probs[guid] \
                                for guid in targetDataset.dev_guid_list])
    em_list = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                        [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(probs_list, em_list)
    global TTDO_SOURCE_PROBS, TTDO_TARGET_PROBS
    for k in source_probs.keys():
        TTDO_SOURCE_PROBS[k] = [source_probs[k]]
    for k in target_probs.keys():
        TTDO_TARGET_PROBS[k] = [target_probs[k]]
    return risks, coverages, auc, sorted_sources, sorted_em    


def maxprob(args, sourceDataset, targetDataset):
    maxprobs_list = np.asarray([sourceDataset.maxprobs[guid] \
                                for guid in sourceDataset.dev_guid_list] + \
                                [targetDataset.maxprobs[guid] \
                                for guid in targetDataset.dev_guid_list])
    em_list = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                        [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(maxprobs_list, em_list)
    global MAXPROB_SOURCE_PROBS, MAXPROB_TARGET_PROBS
    for k in sourceDataset.maxprobs.keys():
        MAXPROB_SOURCE_PROBS[k] = [sourceDataset.maxprobs[k]]
    for k in targetDataset.maxprobs.keys():
        MAXPROB_TARGET_PROBS[k] = [targetDataset.maxprobs[k]]
    """
    plt.plot(coverages, risks, label='Realistic')
    leg = plt.legend(loc=2, prop={'size': 14})
    plt.xlabel('Coverage', fontsize=16)
    plt.ylabel('Risk', fontsize=18)
    plt.title('Risk-Coverage Curve', fontsize=18)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('risk_coverage.png', dpi=400)
    """
    return risks, coverages, auc, sorted_sources, sorted_em


def maxprob_squad_only(args, sourceDataset):
    maxprobs_list = np.asarray([sourceDataset.maxprobs[guid] \
                                for guid in sourceDataset.dev_guid_list])
    em_list = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(maxprobs_list, em_list)
    return risks, coverages, auc, sorted_sources, sorted_em


def maxprob_oracle(args, sourceDataset, exposeDataset, targetDataset):
    x_train = np.asarray([[0, 0, 1, sourceDataset.maxprobs[guid]] \
                            for guid in sourceDataset.train_guid_list] + \
                            [[1, exposeDataset.maxprobs[guid], 0, 0] \
                            for guid in exposeDataset.train_guid_list])
    y_train = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.train_guid_list] + \
                        [exposeDataset.em_dict[guid] \
                            for guid in exposeDataset.train_guid_list])
    x_dev = np.asarray([[0, 0, 1, sourceDataset.maxprobs[guid]] \
                            for guid in sourceDataset.dev_guid_list] + \
                            [[1, targetDataset.maxprobs[guid], 0, 0] \
                            for guid in targetDataset.dev_guid_list])
    y_dev = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                         [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = classifier(args, x_train, y_train, x_dev, y_dev)
    return risks, coverages, auc, sorted_sources, sorted_em


def extrapolate(args, sourceDataset, exposeDataset, targetDataset):
    source_features = sourceDataset.generate_features(args, 'train')
    source_features.update(sourceDataset.generate_features(args, 'dev'))
    if exposeDataset:
        expose_features = exposeDataset.generate_features(args, 'train')
    target_features = targetDataset.generate_features(args, 'dev')

    # Train on source + expose
    x_train = np.asarray([source_features[guid] \
                            for guid in sourceDataset.train_guid_list])
    y_train = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.train_guid_list])
    if exposeDataset and len(exposeDataset.train_guid_list):
        x_train = np.concatenate((x_train, np.asarray([expose_features[guid] \
                                for guid in exposeDataset.train_guid_list])))
        y_train = np.concatenate((y_train, 
                                np.asarray([exposeDataset.em_dict[guid] \
                                for guid in exposeDataset.train_guid_list])))

    # Test on source + target
    x_dev = np.asarray([source_features[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                        [target_features[guid] \
                            for guid in targetDataset.dev_guid_list])
    y_dev = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                         [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = classifier(args, x_train, y_train, x_dev, y_dev, len(sourceDataset.dev_guid_list), list(sourceDataset.dev_guid_list), list(targetDataset.dev_guid_list))
    return risks, coverages, auc, sorted_sources, sorted_em


def extrapolate_oracle(args, sourceDataset, exposeDataset, targetDataset):
    assert exposeDataset
    source_features = sourceDataset.generate_features(args, 'train')
    source_features.update(sourceDataset.generate_features(args, 'dev'))
    expose_features = exposeDataset.generate_features(args, 'train')
    target_features = targetDataset.generate_features(args, 'dev')
    # Train on source + expose
    x_train = np.asarray([source_features[guid] \
                            for guid in sourceDataset.train_guid_list])
    x_train = np.append(x_train, np.array([1]*len(x_train)).reshape(len(x_train), 1), axis=1)
    x_train = np.append(x_train, x_train*0, axis=1)

    x_expose = np.asarray([expose_features[guid] \
                            for guid in exposeDataset.train_guid_list])
    x_expose = np.append(x_expose, np.array([1]*len(x_expose)).reshape(len(x_expose), 1), axis=1)
    x_expose = np.append(x_expose*0, x_expose, axis=1)
    x_train = np.concatenate((x_train, x_expose))
    y_train = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.train_guid_list] + \
                         [exposeDataset.em_dict[guid] \
                            for guid in exposeDataset.train_guid_list])
    # Test on source + target
    x_dev = np.asarray([source_features[guid] \
                            for guid in sourceDataset.dev_guid_list])
    x_dev = np.append(x_dev, np.array([1]*len(x_dev)).reshape(len(x_dev), 1), axis=1)
    x_dev = np.append(x_dev, x_dev*0, axis=1)

    x_target = np.asarray([target_features[guid] \
                            for guid in targetDataset.dev_guid_list])
    x_target = np.append(x_target, np.array([1]*len(x_target)).reshape(len(x_target), 1), axis=1)
    x_target = np.append(x_target*0, x_target, axis=1)
    x_dev = np.concatenate((x_dev, x_target))
    y_dev = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                         [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = classifier(args, x_train, y_train, x_dev, y_dev)
    return risks, coverages, auc, sorted_sources, sorted_em


def outlier_detection(args, sourceDataset, exposeDataset, targetDataset):
    source_features = sourceDataset.generate_features(args, 'train')
    source_features.update(sourceDataset.generate_features(args, 'dev'))
    assert exposeDataset != None
    expose_features = exposeDataset.generate_features(args, 'train')
    target_features = targetDataset.generate_features(args, 'dev')

    # Train on source + expose: outlier detection
    # Note that in-domain is given the label of "1" because 
    # its accuracy is higher than OOD, so this makes it fuzzily
    # predict correctness. So I suppose it's really
    # "inlier detection" :P
    x_train = np.asarray([source_features[guid] \
                            for guid in sourceDataset.train_guid_list])
    y_train = np.asarray([1 for guid in sourceDataset.train_guid_list])
    x_train = np.concatenate((x_train, np.asarray([expose_features[guid] \
                            for guid in exposeDataset.train_guid_list])))
    y_train = np.concatenate((y_train,
                                np.asarray([0 for guid in exposeDataset.train_guid_list])))    

    # Test on source + target: correctness
    x_dev = np.asarray([source_features[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                        [target_features[guid] \
                            for guid in targetDataset.dev_guid_list])
    y_dev = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                         [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list])
    risks, coverages, auc, sorted_sources, sorted_em = classifier(args, x_train, y_train, x_dev, y_dev, len(sourceDataset.dev_guid_list), list(sourceDataset.dev_guid_list), list(targetDataset.dev_guid_list))
    return risks, coverages, auc, sorted_sources, sorted_em



def minimum(args, sourceDataset, targetDataset):
    # Get ideal probabilities (monotonic, non-0)
    dev_len = len(sourceDataset.dev_guid_list) + len(targetDataset.dev_guid_list) \
                + 1
    ideal_probs_list = np.asarray(range(dev_len)[1:])/dev_len 
    em_list = np.asarray([sourceDataset.em_dict[guid] \
                            for guid in sourceDataset.dev_guid_list] + \
                        [targetDataset.em_dict[guid] \
                            for guid in targetDataset.dev_guid_list]) 
    em_list = np.sort(em_list)
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(ideal_probs_list, em_list)

    """
    plt.plot(coverages, risks, label='Optimal')
    leg = plt.legend(loc=2, prop={'size': 14})
    plt.savefig('risk_coverage.png', dpi=400)
    """
    return risks, coverages, auc, sorted_sources, sorted_em


def classifier(args, x_train, y_train, x_dev, y_dev, len_source=0, source_dev_guid_list=None, target_dev_guid_list=None):
    if args.classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators=args.reg2, \
                                     max_depth=args.reg, random_state=0)
    elif args.classifier == 'xgboost':
        clf = xgb.XGBClassifier(objective='binary:logistic', \
                                random_state=0, max_depth=int(args.reg))
    elif args.classifier == 'logreg':
        clf = LogisticRegression(solver='lbfgs', C=args.reg, max_iter=1000)
    elif args.classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=int(args.reg))
    else:
        clf = svm.SVC(gamma='scale', C=args.reg)

    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    dev_score = clf.score(x_dev, y_dev)
    #print("Train accuracy: {}".format(train_score*100))
    #print("Dev accuracy: {}".format(dev_score*100))
    
    probs = [p[1] for p in clf.predict_proba(x_dev)]
    global CAL_SOURCE_PROBS, CAL_TARGET_PROBS
    i, j = 0, 0
    while i < len(source_dev_guid_list):
        if source_dev_guid_list[i] in CAL_SOURCE_PROBS:
            CAL_SOURCE_PROBS[source_dev_guid_list[i]].append(probs[i])
        else:
            CAL_SOURCE_PROBS[source_dev_guid_list[i]] = [probs[i]]
        i += 1
    while j < len(target_dev_guid_list):
        if target_dev_guid_list[j] in CAL_TARGET_PROBS:
            CAL_TARGET_PROBS[target_dev_guid_list[j]].append(probs[i])
        else:
            CAL_TARGET_PROBS[target_dev_guid_list[j]] = [probs[i]]
        i += 1
        j += 1
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(probs, y_dev)
    return risks, coverages, auc, sorted_sources, sorted_em

def evaluate_squad2(mode, targetDataset):
    global MAXPROB_TARGET_PROBS, CAL_TARGET_PROBS
    if mode == 'maxprob':
        na_probs = {k: 1-v[0] for k, v in MAXPROB_TARGET_PROBS.items()}
    else:   
        na_probs = {k: 1-np.mean(v) for k, v in CAL_TARGET_PROBS.items()}

    gold_data = {guid: v for guid, v in targetDataset.gold_data['dev'].items() if guid in targetDataset.dev_guid_list}
    return evaluate(gold_data, targetDataset.preds, na_probs)


def error_analysis(sourceDataset, targetDataset):
    global MAXPROB_TARGET_PROBS, CAL_TARGET_PROBS
    global OVERCONF_LIST, UNDERCONF_LIST
    if MAXPROB_TARGET_PROBS != {}:
        TARGET_POS_PROBS = MAXPROB_TARGET_PROBS
    else:
        TARGET_POS_PROBS = CAL_TARGET_PROBS
    # Choose GUID and the most recent probability,
    # i.e. that of the current split.
    target_tuples = [(guid, prob_list[-1]) for guid, prob_list in TARGET_POS_PROBS.items() if guid in targetDataset.em_dict.keys()]
    target_tuples = sorted(target_tuples, key=lambda x: x[1])
    # Overconfident:
    target_high_conf = target_tuples[-300:]
    target_high_conf = [x for x in target_high_conf if targetDataset.em_dict[x[0]]==0]
    assert len(target_high_conf) >= 20
    np.random.seed(42)
    selected_indices = np.random.choice(list(range(len(target_high_conf))), 20, replace=False)
    j = 0
    for i, example in enumerate(target_high_conf):
        if i not in selected_indices:
            continue
        guid, prob = example
        if guid in OVERCONF_LIST:
            continue
        print("Overconfident {}/20:".format(j+1))
        context = targetDataset.gold_data['dev'][guid]['context']
        question = targetDataset.gold_data['dev'][guid]['question']
        answers = targetDataset.gold_data['dev'][guid]['answers']
        pred = targetDataset.preds[guid]
        utils.render_example(context, question, answers, pred, prob, guid)
        j += 1
        input()
        OVERCONF_LIST.append(guid)

    # Underconfident:
    target_low_conf = target_tuples[:300]
    target_low_conf = [x for x in target_low_conf if targetDataset.em_dict[x[0]]==1]
    assert len(target_low_conf) >= 20
    np.random.seed(42)
    selected_indices = np.random.choice(list(range(len(target_low_conf))), 20, replace=False)
    j = 0
    for i, example in enumerate(target_low_conf):
        if i not in selected_indices:
            continue
        guid, prob = example
        if guid in UNDERCONF_LIST:
            continue
        print("Underconfident {}/20:".format(j+1))
        context = targetDataset.gold_data['dev'][guid]['context']
        question = targetDataset.gold_data['dev'][guid]['question']
        answers = targetDataset.gold_data['dev'][guid]['answers']
        pred = targetDataset.preds[guid]
        utils.render_example(context, question, answers, pred, prob, guid)
        j += 1
        input()
        UNDERCONF_LIST.append(guid)

    return
def train_nn_calibrator(args, sourceDataset, exposeDataset):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    sourceDataset.generate_dl_calib_features(args)



    train_data_list = []
    for key in sourceDataset.train_qa_features.keys():
        train_data_list.append([sourceDataset.train_qa_features[key], sourceDataset.em_dict[key]])
    print("length of squad1.1 split:" + str(len(train_data_list)))
    if exposeDataset:
        exposeDataset.generate_dl_calib_features(args)
        for key in exposeDataset.train_qa_features.keys():
            train_data_list.append([exposeDataset.train_qa_features[key], exposeDataset.em_dict[key]])
    print("length total:" + str(len(train_data_list)))



    all_input_ids = torch.tensor([f[0].input_ids for f in train_data_list], dtype=torch.long)
    all_input_mask = torch.tensor([f[0].input_mask for f in train_data_list], dtype=torch.long)
    all_segment_ids = torch.tensor([f[0].segment_ids for f in train_data_list], dtype=torch.long)
    all_labels = torch.tensor([f[1] for f in train_data_list], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    param_optimizer = list(model.named_parameters())

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    global_step = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_list))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    model.to(device)
    tb_writer = SummaryWriter()

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        loss_total=0
        with tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]) as tepoch:
            for step, batch in enumerate(tepoch):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, labels= batch
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=labels)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps


                loss.backward()
                loss_total += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:

                    optimizer.step()
                    optimizer.zero_grad()


                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)
                    tepoch.set_postfix(loss=loss_total/(step+1))
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    print("saved model to:" + str(output_model_file))
    model_to_save.config.to_json_file(output_config_file)

    # Load a trained model and vocabulary that you have fine-tuned

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.output_dir, 'training_args.bin')
    torch.save(args, output_args_file)

def extrapolate_with_nn_calibrator(args, sourceDataset, targetDataset):
    test_data_list = []
    sourceDataset.generate_dl_calib_features(args)
    targetDataset.generate_dl_calib_features(args)

    all_guids = []
    source_dev_guid_list = []
    target_dev_guid_list = []

    for key in sourceDataset.dev_qa_features.keys():
        test_data_list.append([sourceDataset.dev_qa_features[key], sourceDataset.em_dict[key]])
        all_guids.append(key)
        source_dev_guid_list.append(key)

    for key in targetDataset.dev_qa_features.keys():
        test_data_list.append([targetDataset.dev_qa_features[key], targetDataset.em_dict[key]])
        all_guids.append(key)
        target_dev_guid_list.append(key)
    all_input_ids = torch.tensor([f[0].input_ids for f in test_data_list], dtype=torch.long)
    all_input_mask = torch.tensor([f[0].input_mask for f in test_data_list], dtype=torch.long)
    all_segment_ids = torch.tensor([f[0].segment_ids for f in test_data_list], dtype=torch.long)
    labels = [f[1] for f in test_data_list]
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, shuffle=False, sampler=eval_sampler, batch_size=args.predict_batch_size)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')


    model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=2)
    print("loaded model from:" + args.output_dir)
    model.eval()
    model.to(device)
    probs = []
    for step,batch in enumerate(tqdm(eval_dataloader, desc="Evaluating",
                                                                    disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(device) for t in batch)
        if len(probs) % 1000 == 0:
            logger.info("Processing example: %d" % (len(probs)))
        input_ids, input_mask, segment_ids = batch
        batch_output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        batch_output = torch.nn.functional.softmax(batch_output,dim=-1)
        for i in range(batch_output.shape[0]):
            probs.append(batch_output[i][1].detach().cpu().tolist())
    global CAL_SOURCE_PROBS, CAL_TARGET_PROBS
    i, j = 0, 0
    while i < len(source_dev_guid_list):
        if source_dev_guid_list[i] in CAL_SOURCE_PROBS:
            CAL_SOURCE_PROBS[source_dev_guid_list[i]].append(probs[i])
        else:
            CAL_SOURCE_PROBS[source_dev_guid_list[i]] = [probs[i]]
        i += 1
    while j < len(target_dev_guid_list):
        if target_dev_guid_list[j] in CAL_TARGET_PROBS:
            CAL_TARGET_PROBS[target_dev_guid_list[j]].append(probs[i])
        else:
            CAL_TARGET_PROBS[target_dev_guid_list[j]] = [probs[i]]
        i += 1
        j += 1
    risks, coverages, auc, sorted_sources, sorted_em = utils.get_risk_coverage_info(probs, labels)
    return risks, coverages, auc, sorted_sources, sorted_em

def main():
    args = initialize_args()
    all_risks, all_coverages, all_aucs = [], [], []
    all_sorted_sources, all_sorted_em = [], []
    all_source_maxprobs, all_target_maxprobs = {}, {}
    all_source_em_dict, all_target_em_dict = {}, {}
    all_source_dev_guid_list, all_target_dev_guid_list = [], []

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode in ['minimum', 'maxprob', 'maxprob_squad_only', 'ttdo', 'nn_train', 'nn_eval']:
        total_num_splits = 1
    else:
        total_num_splits = 10
 
    if args.expose_prefix == 'newsqa':
        args.reg2 = 225
    elif args.expose_prefix == 'hotpotqa':
        args.reg2 = 275

    summary_eval_dict = {}

    for split_no in tqdm(range(total_num_splits)):
        if args.task == 'qa':
            sourceDataset = QaDataset(args, 'squad1.1', split_no)
            if args.expose_prefix:
                exposeDataset = QaDataset(args, args.expose_prefix, split_no)
            else:
                exposeDataset = None
            targetDataset = QaDataset(args, args.target_prefix, split_no)
        else:
            raise NotImplementedError("Haven't yet added other tasks.")

        # Collate GUIDs, MaxProbs, EMs for final graphing
        all_source_dev_guid_list.extend(sourceDataset.dev_guid_list)
        all_target_dev_guid_list.extend(targetDataset.dev_guid_list)
        all_source_maxprobs.update(sourceDataset.maxprobs)
        all_target_maxprobs.update(targetDataset.maxprobs)
        all_source_em_dict.update(sourceDataset.em_dict)
        all_target_em_dict.update(targetDataset.em_dict)

        if args.mode == 'ttdo':
            risks, coverages, auc, sorted_sources, sorted_em = test_time_dropout(args, sourceDataset, targetDataset)
        elif args.mode == 'maxprob_squad_only':
            maxprob_squad_only(args, sourceDataset)
            risks, coverages, auc, sorted_sources, sorted_em = maxprob_squad_only(args, sourceDataset)
        elif args.mode == 'maxprob':
            if args.oracle:
                risks, coverages, auc, sorted_sources, sorted_em = maxprob_oracle(args, \
                                        sourceDataset, exposeDataset, targetDataset)
            else:
                #risks, coverages, auc, sorted_sources, sorted_em = minimum(args, sourceDataset, targetDataset)
                #print(auc)
                #pdb.set_trace()
                risks, coverages, auc, sorted_sources, sorted_em = maxprob(args, sourceDataset, targetDataset)
                #print(auc)
        elif args.mode == 'extrapolate':
            if args.oracle:
                risks, coverages, auc, sorted_sources, sorted_em = extrapolate_oracle(args, sourceDataset, \
                                                    exposeDataset, targetDataset)
            else:
                risks, coverages, auc, sorted_sources, sorted_em = extrapolate(args, sourceDataset, \
                                                exposeDataset, targetDataset)
        elif args.mode == 'outlier_detection':
            risks, coverages, auc, sorted_sources, sorted_em = outlier_detection(args, sourceDataset, \
                                                exposeDataset, targetDataset)
        elif args.mode == 'wilcoxon':
            _, _, _, _, _ = maxprob(args, sourceDataset, targetDataset)
            risks, coverages, auc, sorted_sources, sorted_em = extrapolate(args, sourceDataset, \
                                    exposeDataset, targetDataset)
        elif args.mode == 'minimum':
            risks, coverages, auc, sorted_sources, sorted_em = minimum(args, sourceDataset, targetDataset)
        elif args.mode == 'squad2':
            risks, coverages, auc, sorted_sources, sorted_em = maxprob(args, sourceDataset, targetDataset)
            maxprob_eval_dict = evaluate_squad2('maxprob', targetDataset)
            risks, coverages, auc, sorted_sources, sorted_em = extrapolate(args, sourceDataset, \
                                            exposeDataset, targetDataset)
            cal_eval_dict = evaluate_squad2('extrapolate', targetDataset)
            for k, v in cal_eval_dict.items():
                if k in summary_eval_dict:
                    summary_eval_dict[k].append(v)
                else:
                    summary_eval_dict[k] = [v]
        elif args.mode =='nn_train':
            train_nn_calibrator(args, sourceDataset, exposeDataset)
            return
        elif args.mode =='nn_eval':
            risks, coverages, auc, sorted_sources, sorted_em =\
                extrapolate_with_nn_calibrator(args, sourceDataset,targetDataset)
        all_risks.append(risks)
        all_coverages.append(coverages)
        all_aucs.append(auc)
        all_sorted_sources.append(sorted_sources)
        all_sorted_em.append(sorted_em)

        if args.error_analysis:
            error_analysis(sourceDataset, targetDataset)


    avg_risks = np.mean(all_risks, axis=0)
    avg_coverages = np.mean(all_coverages, axis=0)
    avg_auc = np.mean(all_aucs)

    if args.mode == 'squad2':
        print()
        print("Maxprob summmary:")
        print(maxprob_eval_dict)
        print()
        print("Calibrator summary:")
        for k, v in summary_eval_dict.items():
            summary_eval_dict[k] = np.mean(v)
        print(summary_eval_dict)
        print()

    #em_val = sum(list(all_target_em_dict.values())) / len(all_target_em_dict)
    #print("Target EM: {}".format(em_val))
    #em_val = sum(list(all_source_em_dict.values())) / len(all_source_em_dict)
    #print("Source EM: {}".format(em_val))

    print()
    print("AUC: {}".format(100*round(avg_auc, 4)))
   
    index80 = utils.get_coverage_cutoff(avg_risks, 0.8) - 1
    index90 = utils.get_coverage_cutoff(avg_risks, 0.9) - 1
    cov80 = round((100 * avg_coverages[index80]), 4)
    cov90 = round((100 * avg_coverages[index90]), 4)
    print("Coverage at 80% Accuracy = {}".format(cov80)) 
    print("Coverage at 90% Accuracy = {}".format(cov90))

    if args.per_domain:
        utils.get_per_domain_info(all_sorted_sources, all_sorted_em, all_risks)

    if args.mode == 'wilcoxon':
        utils.get_wilcoxon(MAXPROB_SOURCE_PROBS, MAXPROB_TARGET_PROBS, CAL_SOURCE_PROBS, CAL_TARGET_PROBS)
 
    output_source_probs = {}
    output_target_probs = {}

    if args.output_dir:
        if args.mode == 'ttdo':
            output_source_probs.update(TTDO_SOURCE_PROBS)
            output_target_probs.update(TTDO_TARGET_PROBS)
        elif args.mode == 'maxprob':
            output_source_probs.update(MAXPROB_SOURCE_PROBS)
            output_target_probs.update(MAXPROB_TARGET_PROBS)
        else: # extrapolate
            output_source_probs.update(CAL_SOURCE_PROBS)
            output_target_probs.update(CAL_TARGET_PROBS)
        utils.write_output(args, output_source_probs, output_target_probs, 
                                all_source_em_dict, all_target_em_dict, 
                                avg_auc, cov80, cov90)

if __name__ == '__main__':
    main()

