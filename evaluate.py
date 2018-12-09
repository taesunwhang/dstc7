import numpy as np

def evaluate_recall(predictions, target, k_list=[1,2,5,10,50,100]):
    """

    :param predictions: [batch_size, # of candidates]
    :param target: [batch_size]
    :param k_list: [1,2,5,10,50]
    :return: num_correct, num_examples
    """
    #predictions :
    num_examples = np.shape(predictions)[0]
    num_correct = np.zeros([len(k_list)])

    for pred, label in zip(predictions, target):
        for i, k in enumerate(k_list):
            if label in pred[:k]:
                num_correct[i] += 1

    return num_correct, num_examples

def mean_reciprocal_rank(predictions, target):
    """

    :param predictions: [batch_size, # of candidates=100]
    :param target: [batch_size] - index
    :return:
    """
    num_candidates = np.shape(predictions)[1]

    mrr_list = []
    for pred, label in zip(predictions, target):
        #index of target in predictions
        pred_index = np.where(pred == label)
        mrr_list.append(1/(np.int32(pred_index)[0][0]+1))

    tot_sum = sum(mrr_list)

    return tot_sum

def logit_score_recall(prediction_score, target, k_list=[1,2,5,10,50,100]):
    # 1 dialog, 100 response candidates ground truth 1 or 0
    # stack_score = np.stack((prediction_score, target), axis=1)
    stack_score = np.stack((prediction_score, target), axis=1)
    pos_score = prediction_score[0]
    sort_pred_score = sorted(stack_score, key = lambda x:x[0], reverse=True)

    num_correct = np.zeros([len(k_list)])

    for p_i, sorted_score in enumerate(sort_pred_score):
        if sorted_score[1] == 1:
            pos_index = p_i

    # print(pos_index + 1)
    for i, k in enumerate(k_list):
        if pos_index + 1  <= k:
            num_correct[i] += 1

    return num_correct

def logit_mean_reciprocal_rank(prediction_score, target):
    stack_score = np.stack((prediction_score, target), axis=1)
    # pos_score = prediction_score[0]
    sort_pred_score = sorted(stack_score, key = lambda x:x[0],  reverse=True)

    mrr = 0
    for i, sorted_score in enumerate(sort_pred_score):
        #positive index
        if sorted_score[1] == 1:
            mrr = 1 / (i+1)
            break

    return mrr