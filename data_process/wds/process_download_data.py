import os
import webdataset as wds
import argparse

def get_sample_num(src_pth):
    dataset = wds.WebDataset(src_pth)
    data_num = 0
    for _ in dataset:
        data_num += 1
    return  data_num

def split_download_shards(src_fold, res_fold, tarproc_fold):
    fn_list = os.listdir(src_fold)
    fn_list = [fn for fn in fn_list if ".tar" in fn]
    fn_list.sort()
    print('downloaded data has {} shards'.format(len(fn_list)))
    fn_idx = 0
    for fn in fn_list:
        src_pth = src_fold + "/" + fn
        res_pth = res_fold + "/" + '{}'.format(fn_idx)  # output shard_name will be [fn_idx]-[part_idx].tar
        cmd = "python {}/tarsplit -n {} -o {} {} \n".format(tarproc_fold, 1000, res_pth, src_pth)
        os.system(cmd)
        fn_idx += 1

def combine_shards(res_fold, tarproc_fold):
    fn_list = os.listdir(res_fold)
    # ---find the last part of each fn_idx
    fn_2_lastpart = {}
    for fn in fn_list:
        print(fn)
        part_idx = int(fn.split('.')[0].split('-')[1])
        fn_idx = fn.split('-')[0]

        if fn_idx not in fn_2_lastpart:
            fn_2_lastpart[fn_idx] = (part_idx, fn)
        else:
            if part_idx > fn_2_lastpart[fn_idx][0]:
                fn_2_lastpart[fn_idx] = (part_idx, fn)

    lastpart_list = [ele[1] for ele in list(fn_2_lastpart.values())] #remove part_idx

    # generate cmd for combining tar files.
    cmd = 'python {}/tarcats'.format(tarproc_fold)
    combin_tar_pth = res_fold + '/combine.tar'
    for fn in lastpart_list:
        cmd += ' ' + res_fold + '/' + fn
    cmd += ' -o {}'.format(combin_tar_pth)

    print(cmd)
    os.system(cmd)

    return lastpart_list, combin_tar_pth

def rename_shard(shard_fold):
    fn_list = os.listdir(shard_fold)
    for idx, fn in enumerate(fn_list):
        src_pth = shard_fold + '/' + fn
        tgt_pth = shard_fold + '/' + '{}'.format(idx).zfill(5) + '.tar'
        os.rename(src_pth, tgt_pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--src_fold', required=True, type=str)
    parser.add_argument('--res_fold', required=True, type=str)
    parser.add_argument('--tarproc_fold', required=True, type=str)

    args = parser.parse_args()

    src_fold = args.src_fold
    res_fold = args.res_fold
    tarproc_fold = args.tarproc_fold
    #tarproc

    print(src_fold, res_fold)
    #---1. split shards generated by image2dataset to smaller shards, each of which has 1000 samples.
    split_download_shards(src_fold, res_fold, tarproc_fold)


    #----2. the last part of each fn_idx may have less than 1000 samples, so we combine them to one shard.
    lastpart_list, combin_tar_pth = combine_shards(res_fold, tarproc_fold)

    #---3. split the combined shard to shards with 1000 samples again.
    print('split the combined shard...')
    src_pth = combin_tar_pth
    res_pth = res_fold + '/combine'
    cmd = "python {}/tarsplit -n {} -o {} {} \n".format(tarproc_fold, 1000, res_pth, src_pth)
    os.system(cmd)


    #delte the shards that have been combined.
    for ele in lastpart_list:
        os.remove(res_fold + '/' + ele)
    os.remove(combin_tar_pth)

    #rename shards
    rename_shard(res_fold)
