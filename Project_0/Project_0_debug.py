#!/usr/bin/env python3

def get_sum_metrics(predictions, metrics=[]):
    for i in range(3):
        # c = i
        metrics.append(lambda x, i=i: x + i)
        # if i==0:
        #     metrics.append(lambda x: x + 0)
        # elif i==1:
        #     metrics.append(lambda x: x + 1)
        # elif i==2:
        #     metrics.append(lambda x: x + 2)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics

# breakpoint()
# predictions = 0
# get_sum_metrics(predictions, metrics=[])
# import pdb; pdb.set_trace()