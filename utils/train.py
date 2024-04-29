"""Training utils."""
import datetime
import os

LOGS_PATH = "\\logs"


def get_savedir(model, dataset):
    """Get unique saving directory name."""
    print(LOGS_PATH)
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(LOGS_PATH, date, dataset, model + dt.strftime('_%H_%M_%S'))
    os.makedirs(save_dir)
    return save_dir


def avg_both(mrs, mrrs, hits, amris, mr_deviations):
    """Aggregate metrics for missing lhs and rhs.

    Args:
        mrs: Dict[str, float]
        mrrs: Dict[str, float]
        hits: Dict[str, torch.FloatTensor]
        amris: Dict[str, float]
        mr_deviations: Dict[str, float]
    Returns:
        Dict[str, torch.FloatTensor] mapping metric name to averaged score
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.

    # added AMRI and MR_deviation
    amri = (amris['lhs'] + amris['rhs']) / 2.
    mr_deviation = (mr_deviations['lhs'] + mr_deviations['rhs']) / 2.

    return {'MR': mr, 'MRR': mrr, 'hits@[1,3,10]': h, 'AMRI': amri, 'MR_deviation': mr_deviation}


def format_metrics(metrics, split):
    """Format metrics for logging."""
    result = "\t {} MR: {:.2f} | ".format(split, metrics['MR'])
    result += "MRR: {:.3f} | ".format(metrics['MRR'])
    result += "H@1: {:.3f} | ".format(metrics['hits@[1,3,10]'][0])
    result += "H@3: {:.3f} | ".format(metrics['hits@[1,3,10]'][1])
    result += "H@10: {:.3f} | ".format(metrics['hits@[1,3,10]'][2])

    # added AMRI and MR_deviation
    result += "AMRI: {:.6f} | ".format(metrics['AMRI'])
    result += "MR_deviation: {:.3f}".format(metrics['MR_deviation'])
    return result


def write_metrics(writer, step, metrics, split):
    """Write metrics to tensorboard logs."""
    writer.add_scalar('{}_MR'.format(split), metrics['MR'], global_step=step)
    writer.add_scalar('{}_MRR'.format(split), metrics['MRR'], global_step=step)
    writer.add_scalar('{}_H1'.format(split), metrics['hits@[1,3,10]'][0], global_step=step)
    writer.add_scalar('{}_H3'.format(split), metrics['hits@[1,3,10]'][1], global_step=step)
    writer.add_scalar('{}_H10'.format(split), metrics['hits@[1,3,10]'][2], global_step=step)

    # added AMRI and MR_deviation
    writer.add_scalar('{}_AMRI'.format(split), metrics['AMRI'], global_step=step)
    writer.add_scalar('{}_MR_deviation'.format(split), metrics['MR_deviation'], global_step=step)


def count_params(model):
    """Count total number of trainable parameters in model"""
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total