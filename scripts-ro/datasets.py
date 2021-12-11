"""
Dataset classes.

"""

import os
import pdb
import utils
import numpy as np
import random
from src.bert_squad_dataset_utils import read_mrqa_examples,  convert_examples_to_features
from pytorch_pretrained_bert.tokenization import BertTokenizer
class Dataset(object):
    def _calculate_maxprobs(self):
        """
        Calculate the maxprobs from nbest data.
        Args:
            None
        Returns:
            a dict of {QID: maxprob}
        """
        raise NotImplementedError

    def _generate_predictions(self):
        """
        Generate predictions from nbest data.
        Predictions are non-null.
        Args:
            None
        Returns:
            a dict of {QID: "pred"}
        """
        raise NotImplementedError

    def _calculate_em(self):
        """
        Calculate the EM dictionary from nbest data.
        Args:
            None
        Returns:
            a dict of {QID: 1[EM]}
        """
        raise NotImplementedError

    def get_features(self):
        """
        Generate dataset features.
        Args:
            None
        Returns:
            A dict of {QID: [features]}
        """
        raise NotImplementedError

    def precompute_all(self, nbest_flag=True, \
                        substring_flag=False, hidden_flag=False):
        """
        Precompute nbest, substring and hidden
        features of the model for the dataset,
        per flags.
        Stores JSONs in self.model_dir.
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError



class QaDataset(Dataset):
    def __init__(self, args, dataset_prefix, split_no):
        self.train_qa_features = None
        self.dev_qa_features = None
        gold_data_train_path = \
            'data_splits/{}_train_split_{}.jsonl'.format(\
            dataset_prefix, split_no)
        self.gold_data_train_path = gold_data_train_path
        self.gold_data = {'train': {}, 'dev': {}}

        if args.mode not in ['minimum', 'maxprob', 'maxprob_squad_only', 'ttdo']:
            self.gold_data['train'] = utils.read_gold_data(args.task, gold_data_train_path,\
                dataset_prefix)

        if args.mode == 'extrapolate' and not args.expose_prefix \
            and dataset_prefix=='squad1.1':
            # Need to train calibrator on double
            # the usual amount of SQuAD examples
            # one split already read, above.
            self._get_double_squad(args, dataset_prefix, split_no)

        gold_data_dev_path = \
            'data_splits/{}_test_split_{}.jsonl'.format(\
            dataset_prefix, 0)
        self.gold_data_dev_path = gold_data_dev_path
        self.gold_data['dev'] = utils.read_gold_data(args.task, gold_data_dev_path,\
            dataset_prefix)


        percentage = 1.0
        if args.fraction_id:
            if 'squad' in dataset_prefix: 
                percentage = args.fraction_id
            else:
                percentage = 1 - args.fraction_id

        num_train = int(percentage * len(self.gold_data['train'].keys()))
        
        self.train_guid_list = list(self.gold_data['train'].keys())[:num_train]

        if dataset_prefix == 'squad2.0':
            np.random.seed(42)
            self.dev_guid_list = np.random.choice(list(self.gold_data['dev'].keys()), 4000, replace=False)
            unanswerable_guids = [guid for guid in self.gold_data['dev'].keys() if self.gold_data['dev'][guid]['answers'][0]==""]
            #print("Fraction of unanswerable questions in the entire dataset = {}".format(float(len(unanswerable_guids) / len(self.gold_data['dev'].keys()))))
            unanswerable_guids = [guid for guid in unanswerable_guids if guid in self.dev_guid_list]
            #print("Fraction of unanswerable questions in the selected dev dataset = {}".format(float(len(unanswerable_guids) / len(self.dev_guid_list))))
            # If you only want to evaluate on unanswerable questions:
            # unanswerable_guids = [guid for guid in self.gold_data['dev'].keys() if self.gold_data['dev'][guid]['answers'][0]==""]
            # assert len(unanswerable_guids) >= 4000
            # self.dev_guid_list = np.random.choice(unanswerable_guids, 4000, replace=False)
        else:
            num_dev = int(percentage * len(self.gold_data['dev'].keys()))
            self.dev_guid_list = list(self.gold_data['dev'].keys())[:num_dev]

        nbest_path = os.path.join(args.model_dir, \
                                    '{}-nbest_predictions.json'.format\
                                    (dataset_prefix))
        try:
            self.nbest_data = utils.read_nbest_data(nbest_path, self.dev_guid_list)
        except:
            self.nbest_data = utils.read_nbest_data_from_long(nbest_path+'l', self.dev_guid_list)


        if args.mode in ['minimum', 'maxprob', 'maxprob_squad_only', 'ttdo']:
            pass
        elif dataset_prefix != 'squad1.1' and dataset_prefix != 'squad2.0':
            nbest_path = os.path.join(args.model_dir, \
                                    '{}_train-nbest_predictions.jsonl'.format\
                                    (dataset_prefix))
            self.nbest_data.update(utils.read_nbest_data_from_long(nbest_path, \
                                        self.train_guid_list))
        elif dataset_prefix != 'squad2.0':
            try:
                self.nbest_data.update(utils.read_nbest_data(nbest_path, self.train_guid_list))
            except:
                self.nbest_data.update(utils.read_nbest_data_from_long(nbest_path+'l', self.train_guid_list))

        # call calc maxprob
        self.maxprobs = self._calc_maxprobs()
        self.second_maxprobs = self._calc_maxprobs(2)
        self.third_maxprobs = self._calc_maxprobs(3)
        self.fourth_maxprobs = self._calc_maxprobs(4)
        self.fifth_maxprobs = self._calc_maxprobs(5)
        self.dataset_prefix = dataset_prefix

        # call gen preds
        self.preds = self._generate_predictions()
        if args.ttdo_calibrator:
            self.more_preds = [self._generate_more_predictions(i) for i in range(0,5)]
        # call calc EM
        if args.strict_eval:
            self.em_dict = self._calc_em_strict('train')
            self.em_dict.update(self._calc_em_strict('dev'))
        else:
            self.em_dict = self._calc_em('train')
            self.em_dict.update(self._calc_em('dev'))

    def _calc_maxprobs(self, i=1):
        maxprobs = {}
        for k, v in self.nbest_data.items():
            try:
                if v[0]['text'] != "":
                    maxprobs[k] = float(v[0+i-1]['probability'])
                else:
                    maxprobs[k] = float(v[1+i-1]['probability'])
            except:
                maxprobs[k] = 0
        return maxprobs


    def _generate_predictions(self):
        preds = {k: v[0]['text'] if v[0]['text']!=""
                    else v[1]['text']
                    for k, v in self.nbest_data.items()}
        return preds

    def _generate_more_predictions(self, i):
        i_preds = {}
        for k, v in self.nbest_data.items():
            try:
                if v[0]['text']!="":
                    i_preds[k] = v[i]['text']
                else:
                    i_preds[k] = v[i+1]['text']
            except:
                if v[0]['text']!="":
                    i_preds[k] = v[-1]['text']
                else:
                    i_preds[k] = v[-1]['text']
        return i_preds

    def _calc_em(self, train_or_dev):
        em_dict = {}
        if train_or_dev == 'train':
            guid_list = self.train_guid_list
        else:
            guid_list = self.dev_guid_list
        for guid in guid_list:
            m = max(utils.exact_match_score(x, self.preds[guid]) \
                        for x in self.gold_data[train_or_dev][guid]['answers'])
            em_dict[guid] = 1 if m else 0
        return em_dict

    def _calc_em_strict(self, train_or_dev):
        # Only consider the first answer correct
        em_dict = {}
        if train_or_dev == 'train':
            guid_list = self.train_guid_list
        else:
            guid_list = self.dev_guid_list
        for guid in guid_list:
            m = utils.exact_match_score(self.gold_data[train_or_dev][guid]['answers'][0], self.preds[guid])
            em_dict[guid] = 1 if m else 0
        return em_dict

    def _get_double_squad(self, args, dataset_prefix, split_no):
        gold_data_train_path = \
                'data_splits/{}_train_split_{}.jsonl'.format(\
                dataset_prefix, (split_no+1)%10)
        self.gold_data['train'].update(utils.read_gold_data(args.task, gold_data_train_path,\
            dataset_prefix))
        gold_data_train_path = \
            'data_splits/{}_train_split_{}.jsonl'.format(\
            dataset_prefix, (split_no+2)%10)
        self.gold_data['train'].update(utils.read_gold_data(args.task, gold_data_train_path,\
            dataset_prefix))
        gold_data_train_path = \
            'data_splits/{}_train_split_{}.jsonl'.format(\
            dataset_prefix, (split_no+3)%10)
        self.gold_data['train'].update(utils.read_gold_data(args.task, gold_data_train_path,\
            dataset_prefix))
        np.random.seed(42)
        chosen_train = np.random.choice(list(self.gold_data['train'].keys()), 3200, replace=False)
        self.gold_data['train'] = {k: v for k, v in self.gold_data['train'].items() if k in chosen_train}
        
    def generate_features(self, args, train_or_dev):
        if train_or_dev == 'train':
            guid_list = self.train_guid_list
        else:
            guid_list = self.dev_guid_list

        if args.ttdo_calibrator:
            mean_dict_list, var_dict_list = \
                    utils.get_more_prob_stats_test(args, guid_list, \
                                self.more_preds, self.dataset_prefix, \
                                train_or_dev)

        features = {}
        for guid in guid_list:
            features[guid] = []
            if args.ablate != 'context_len':
                features[guid].append(len( \
                                self.gold_data[train_or_dev][guid]['context'].split()))
            if not args.ttdo_calibrator:
                if args.ablate != 'all_prob':
                    if args.ablate != 'maxprob':
                        features[guid].append(self.maxprobs[guid])
                    if args.ablate != 'other_prob':
                        features[guid].append(self.second_maxprobs[guid])
                        features[guid].append(self.third_maxprobs[guid])
                        features[guid].append(self.fourth_maxprobs[guid])
                        features[guid].append(self.fifth_maxprobs[guid])
            if args.ablate != 'pred_len':
                features[guid].append(len(self.preds[guid].split()))
            if args.ttdo_calibrator:
                for i in range(5):
                    features[guid].append(mean_dict_list[i][guid])
                    features[guid].append(var_dict_list[i][guid])
        return features

    def generate_dl_calib_features(self, args):
        #need to USE STRICT_EVAL

        train_mrqa_examples = read_mrqa_examples(self.gold_data_train_path, True)
        dev_mrqa_examples = read_mrqa_examples(self.gold_data_dev_path, True)


        for ex in train_mrqa_examples:
            ex.question_text = ex.question_text + " Answer: \""+self.preds[ex.qas_id] + "\""
        for ex in dev_mrqa_examples:
            ex.question_text = ex.question_text + " Answer: \""+self.preds[ex.qas_id]+ "\""

        # for i in range(20):
        #     print(train_mrqa_examples[i].doc_tokens)
        #     print(train_mrqa_examples[i].question_text)
        #
        #     print(self.preds[train_mrqa_examples[i].qas_id])
        #     print(self.gold_data['train'][train_mrqa_examples[i].qas_id]['answers'])
        #     print(train_mrqa_examples[i].orig_answer_text)
        #     print(self.em_dict[train_mrqa_examples[i].qas_id])
        #     print()
        #
        # print_count=0
        # total_incorrect_count = 0
        # for i,ex in enumerate(train_mrqa_examples):
        #     if not self.em_dict[ex.qas_id]:
        #         total_incorrect_count+=1
        #         if print_count<20:
        #             print(train_mrqa_examples[i].doc_tokens)
        #             print(train_mrqa_examples[i].question_text)
        #
        #             print(self.preds[train_mrqa_examples[i].qas_id])
        #             print(self.gold_data['train'][train_mrqa_examples[i].qas_id]['answers'])
        #             print(train_mrqa_examples[i].orig_answer_text)
        #             print(self.em_dict[train_mrqa_examples[i].qas_id])
        #             print()
        #             print_count+=1
        # print(total_incorrect_count)
        # print(len(train_mrqa_examples))

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        train_mrqa_features = convert_examples_to_features(
            examples=train_mrqa_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True, include_qa_id=True)

        dev_mrqa_features= convert_examples_to_features(
            examples=dev_mrqa_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True, include_qa_id=True)
        key_set = []
        # for i in range(20):
        #     print(train_mrqa_examples[i].qas_id)
        #     key_set.append(train_mrqa_examples[i].qas_id)
        #     print(train_mrqa_features[train_mrqa_examples[i].qas_id].input_ids)
        #     print(convert_examples_to_features([train_mrqa_examples[i]],
        #                                        tokenizer=tokenizer,
        #                                        max_seq_length=args.max_seq_length,
        #                                        doc_stride=args.doc_stride,
        #                                        max_query_length=args.max_query_length,
        #                                        is_training=True, include_qa_id=True
        #                                        )[train_mrqa_examples[i].qas_id].input_ids)
        self.train_qa_features = train_mrqa_features
        self.dev_qa_features = dev_mrqa_features


