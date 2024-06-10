"""Training utils."""
import datetime
import os

import wandb

from ensemble import Constants

LOGS_PATH = "logs"


def get_savedir(model, dataset):
    """Get unique saving directory name."""
    print(LOGS_PATH)
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(LOGS_PATH, date, dataset, model + dt.strftime('_%H_%M_%S'))
    os.makedirs(save_dir)
    return save_dir


def avg_both(mrs, mrrs, hits, amris, rank_deviations, epoch=None):
    """Aggregate metrics for missing lhs and rhs.

    Args:
        mrs: Dict[str, float]
        mrrs: Dict[str, float]
        hits: Dict[str, torch.FloatTensor]
        amris: Dict[str, float]
        rank_deviations: Dict[str, float]
    Returns:
        Dict[str, torch.FloatTensor] mapping metric name to averaged score
    """
    mrs['average'] = (mrs['lhs'] + mrs['rhs']) / 2.
    mrrs['average'] = (mrrs['lhs'] + mrrs['rhs']) / 2.
    hits['average'] = (hits['lhs'] + hits['rhs']) / 2.

    # added AMRI and rank_deviation
    amris['average'] = (amris['lhs'] + amris['rhs']) / 2.
    rank_deviations['average'] = (rank_deviations['lhs'] + rank_deviations['rhs']) / 2.
    hits1_wandb = {'average': hits['average'][0], 'lhs': hits['lhs'][0], 'rhs': hits['rhs'][0]}
    hits3_wandb = {'average': hits['average'][1], 'lhs': hits['lhs'][1], 'rhs': hits['rhs'][1]}
    hits10_wandb = {'average': hits['average'][2], 'lhs': hits['lhs'][2], 'rhs': hits['rhs'][2]}

    if Constants.LOG_WANDB:
        # Online logging for metrics
        wandb.log({'MR': mrs, 'MRR': mrrs, 'HITS@1': hits1_wandb, 'HITS@3': hits3_wandb, 'HITS@10': hits10_wandb,
                   'AMRI': amris, 'rank_deviation': rank_deviations, 'epoch': epoch})

    return {'MR': mrs, 'MRR': mrrs, 'hits@[1,3,10]': hits, 'AMRI': amris, 'rank_deviation': rank_deviations}


def format_metrics(metrics, split):
    """Format metrics for logging."""
    result = "\n"
    for mode in ['average', 'rhs', 'lhs']:
        result += f"{mode} {split} metrics:\tMR: {metrics['MR'][mode]:.2f} | "
        result += f"MRR: {metrics['MRR'][mode]:.3f} | "
        result += f"H@1: {metrics['hits@[1,3,10]'][mode][0]:.3f} | "
        result += f"H@3: {metrics['hits@[1,3,10]'][mode][1]:.3f} | "
        result += f"H@10: {metrics['hits@[1,3,10]'][mode][2]:.3f} | "

        # added AMRI and rank_deviation
        result += f"AMRI: {metrics['AMRI'][mode]:.3f} | "
        result += f"rank_deviation: {metrics['rank_deviation'][mode]:.3f}\n"

    return result.rstrip()


def write_metrics(writer, step, metrics, split):
    """Write metrics to tensorboard logs."""
    writer.add_scalar('{}_MR'.format(split), metrics['MR'], global_step=step)
    writer.add_scalar('{}_MRR'.format(split), metrics['MRR'], global_step=step)
    writer.add_scalar('{}_H1'.format(split), metrics['hits@[1,3,10]'][0], global_step=step)
    writer.add_scalar('{}_H3'.format(split), metrics['hits@[1,3,10]'][1], global_step=step)
    writer.add_scalar('{}_H10'.format(split), metrics['hits@[1,3,10]'][2], global_step=step)

    # added AMRI and rank_deviation
    writer.add_scalar('{}_AMRI'.format(split), metrics['AMRI'], global_step=step)
    writer.add_scalar('{}_rank_deviation'.format(split), metrics['rank_deviation'], global_step=step)


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
