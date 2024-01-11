import os.path
import os
import argparse


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Prepare the image-data set for training or prediction")
    argparser.add_argument('--logs_path', type=str, required=False,
                           default="/dccstor/alfassy/dev/open_clip_honda/Outputs",
                           help="Path to logs saved during training")
    argparser.add_argument('--run_name', type=str, required=False, default="nce_many_2",
                           help="Name of training run (run folder)")
    argparser.add_argument('--exp_mode', type=str, default="many",
                           help="exp_mode used in training. Options: many/few/one/zero.")
    args = argparser.parse_args()
    return args


def gather_results(logs_path):
    folds = [0, 1, 2, 3, 4]
    # folds = [0, 1, 3, 4]
    # folds = [2]
    # inner_folds = [0, 1, 2, 3, 4]
    inner_folds = [0]
    t2i_acc1, t2i_acc5, t2i_acc10 = [], [], []
    i2t_acc1, i2t_acc5, i2t_acc10 = [], [], []
    epoch = 2
    active = False
    for i in folds:
        for j in inner_folds:
            log_path = os.path.join(logs_path, f'rep_nce_zero_{i}', 'out.log')
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r') as file:
                for line in file:
                    line = line.rstrip()
                    if active:
                        if 'Txt to img acc' in line:
                            splits = line.split(' ')
                            t2i_acc1.append(float(splits[-3].split(',')[0]))
                            t2i_acc5.append(float(splits[-2].split(',')[0]))
                            t2i_acc10.append(float(splits[-1].split(',')[0]))
                        elif 'Img to txt acc' in line:
                            splits = line.split(' ')
                            i2t_acc1.append(float(splits[-3].split(',')[0]))
                            i2t_acc5.append(float(splits[-2].split(',')[0]))
                            i2t_acc10.append(float(splits[-1].split(',')[0]))
                            active = False
                            break
                    else:
                        if f'Start epoch {epoch}' in line:
                        # if f'start running Image/Text Retrieval evaluation' in line:
                            active = True
                            print(log_path)

    print(f'found {len(t2i_acc1)} folds results')
    print(f't2i acc 1,5,10: {t2i_acc1}, {t2i_acc5}, {t2i_acc10}')
    print(f'i2t acc 1,5,10: {i2t_acc1}, {i2t_acc5}, {i2t_acc10}')
    t2i_acc1_average = sum(t2i_acc1) / len(t2i_acc1)
    t2i_acc5_average = sum(t2i_acc5) / len(t2i_acc5)
    t2i_acc10_average = sum(t2i_acc10) / len(t2i_acc10)
    i2t_acc1_average = sum(i2t_acc1) / len(i2t_acc1)
    i2t_acc5_average = sum(i2t_acc5) / len(i2t_acc5)
    i2t_acc10_average = sum(i2t_acc10) / len(i2t_acc10)
    print(f't2i average 1,5,10: {t2i_acc1_average}, {t2i_acc5_average}, {t2i_acc10_average}')
    print(f'i2t average 1,5,10: {i2t_acc1_average}, {i2t_acc5_average}, {i2t_acc10_average}')
    # t2i_acc1_median =statistics.median(t2i_acc1)
    # t2i_acc5_median = statistics.median(t2i_acc5)
    # t2i_acc10_median =statistics.median(t2i_acc10)
    # i2t_acc1_median = statistics.median(i2t_acc1)
    # i2t_acc5_median = statistics.median(i2t_acc5)
    # i2t_acc10_median =statistics.median(i2t_acc10)
    # print(f't2i median 1,5,10: {t2i_acc1_median}, {t2i_acc5_median}, {t2i_acc10_median}')
    # print(f'i2t median 1,5,10: {i2t_acc1_median}, {i2t_acc5_median}, {i2t_acc10_median}')


def gather_results_train_test(logs_path, run_name, exp_mode):
    if exp_mode == 'zero':
        folds = [0, 1, 2, 3, 4]
        epoch = 2
    elif exp_mode == 'one':
        folds = [0, 1, 2, 3, 4]
        epoch = 19
    elif exp_mode == 'few':
        folds = [0, 1, 3, 4]
        epoch = 19
    elif exp_mode == 'many':
        folds = [2]
        epoch = 19
    elif exp_mode == 'ikea':
        folds = [0, 1, 2, 3, 4]
        epoch = 19
    else:
        raise NotImplementedError(f'{exp_mode} not supported')
    if 'clip' in run_name.lower() or 'test' in run_name:
        trigger_line = f'start running Image/Text Retrieval evaluation'
    else:
        trigger_line = f'Start epoch {epoch}'
    t2i_acc1, t2i_acc5, t2i_acc10 = [], [], []
    i2t_acc1, i2t_acc5, i2t_acc10 = [], [], []
    active = False
    for fold in folds:
        log_path = os.path.join(logs_path, f'{run_name}_{fold}/out.log')
        if not os.path.exists(log_path):
            continue
        with open(log_path, 'r') as file:
            for line in file:
                line = line.rstrip()
                if active:
                    if 'Txt to img acc' in line:
                        splits = line.split(' ')
                        t2i_acc1.append(float(splits[-3].split(',')[0]))
                        t2i_acc5.append(float(splits[-2].split(',')[0]))
                        t2i_acc10.append(float(splits[-1].split(',')[0]))
                    elif 'Img to txt acc' in line:
                        splits = line.split(' ')
                        i2t_acc1.append(float(splits[-3].split(',')[0]))
                        i2t_acc5.append(float(splits[-2].split(',')[0]))
                        i2t_acc10.append(float(splits[-1].split(',')[0]))
                        active = False
                        break
                else:
                    if trigger_line in line:
                        active = True
                        print(log_path)
    print(f'found {len(t2i_acc1)} folds results')
    t2i_acc1_average = sum(t2i_acc1) / len(t2i_acc1)
    t2i_acc5_average = sum(t2i_acc5) / len(t2i_acc5)
    t2i_acc10_average = sum(t2i_acc10) / len(t2i_acc10)
    i2t_acc1_average = sum(i2t_acc1) / len(i2t_acc1)
    i2t_acc5_average = sum(i2t_acc5) / len(i2t_acc5)
    i2t_acc10_average = sum(i2t_acc10) / len(i2t_acc10)
    print(f't2i acc 1,5,10: {t2i_acc1_average}, {t2i_acc5_average}, {t2i_acc10_average}')
    print(f'i2t acc 1,5,10: {i2t_acc1_average}, {i2t_acc5_average}, {i2t_acc10_average}')


def check_existing_results(log_path, total_epochs):
    epoch = total_epochs - 1
    active = False
    log_path = log_path
    if not os.path.exists(log_path):
        return False
    with open(log_path, 'r') as file:
        for line in file:
            line = line.rstrip()
            if active:
                if 'Txt to img acc' in line:
                    return True
            else:
                if f'Start epoch {epoch}' in line:
                    active = True


if __name__ == "__main__":
    # gather_results('/dccstor/alfassy/dev/open_clip_honda/Outputs')
    args = parse_arguments()
    gather_results_train_test(args.logs_path, args.run_name, args.exp_mode)
