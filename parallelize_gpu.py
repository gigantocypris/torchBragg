import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import h5py

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from vae.model import AutoEncoder
from thirdparty.adamax import Adamax
import vae.utils as utils
import vae.datasets as datasets

import matplotlib.pyplot as plt

import wandb

def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # Get data loaders.
    train_queue = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model = AutoEncoder(args, writer, arch_instance) # input are the known model parameters, loss is the probability distribution of experimental data
    model = model.cuda()


    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        adamax_fn = Adamax
    else:
        adamax_fn = torch.optim.Adamax

    cnn_optimizer = adamax_fn(model.parameters(), args.learning_rate,
                            weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)

    grad_scalar = GradScaler(2**10)

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt') # best checkpoint is saved here
    train_nelbo_vec = [min_train_nelbo]

    # check if checkpoint file exists
    if os.path.isfile(checkpoint_file) and args.cont_training:
        logging.info('loading the model.')

        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        train_nelbo_vec = checkpoint['train_nelbo_vec']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
        min_train_nelbo = checkpoint['min_train_nelbo']
        epoch_min_train_nelbo = checkpoint['epoch_min_train_nelbo']
    else:
        global_step, init_epoch = 0, 0

    epoch = init_epoch

    
    steepness = -np.log((1-args.pnm_fraction)/args.pnm_fraction)/args.pnm_warmup_epochs
    # Initial pnm_implement, calculated here in case of args.epochs==0
    args.pnm_implement = (2 / (1 + np.exp(-steepness*epoch)) - 1.0)*(args.pnm-args.pnm_start) + args.pnm_start

    
    for epoch in range(init_epoch, args.epochs):

        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)


        # Logging.
        logging.info('epoch %d', epoch)
        args.pnm_implement = (2 / (1 + np.exp(-steepness*epoch)) - 1.0)*(args.pnm-args.pnm_start) + args.pnm_start

        logging.info('pnm_implement %d', args.pnm_implement)

        # Training.
        train_nelbo, global_step = train(args, train_queue, model, model_ring, cnn_optimizer, cnn_optimizer_ring,
                                         grad_scalar, global_step, warmup_iters, writer, logging,
                                        )
        

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        logging.info('train_nelbo %f', train_nelbo)
        writer.add_scalar('train/nelbo', train_nelbo, global_step)
        if args.log_wandb:
            wandb.log({"train_nelbo": train_nelbo})
        train_loss_vec.append(train_loss)

        if args.global_rank == 0:
            checkpoint_file = os.path.join(args.save, 'checkpoint.pt')

            logging.info('saving the model in ' + checkpoint_file)
            
            save_dict = {'epoch': epoch + 1, 
                            'train_nelbo_vec': train_nelbo_vec,
                            'state_dict': model.state_dict(),
                            'optimizer': cnn_optimizer.state_dict(), 
                            'global_step': global_step,
                            'args': args, 'arch_instance': arch_instance, 
                            'scheduler': cnn_scheduler.state_dict(),
                            'grad_scalar': grad_scalar.state_dict()}

            torch.save(save_dict, checkpoint_file)
        

    writer.close()




def get_loss(x, args, model, model_ring, theta,
                           sparse_sinogram_raw, x_size, object_id):
    dist = model(x)
    recon_loss = dist.log_prob(experimental_data)
    return(recon_loss)


def train(args, train_queue, model, model_ring, cnn_optimizer, cnn_optimizer_ring, grad_scalar, 
          global_step, warmup_iters, writer, logging,
          ):

    model.train()

    for step, x_full in enumerate(train_queue):
        # x_full is (orientation, experimental_img)
        orientation, experimental_img = x_full

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr
            if args.model_ring_artifact:
                for param_group in cnn_optimizer_ring.param_groups:
                    param_group['lr'] = lr
        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)
            if args.model_ring_artifact:
                utils.average_params(model_ring.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        if args.model_ring_artifact:
            cnn_optimizer_ring.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag, \
            logits_ring, log_q_ring, log_p_ring, kl_all_ring, kl_diag_ring, \
            sino_raw_dist, phantom, recon_loss = \
                process_decoder_output(x, args, model, model_ring, theta, sparse_sinogram_raw, x_size, object_id)
            
            loss, norm_loss, bn_loss, wdn_coeff, kl_coeff, kl_coeffs, kl_vals = calculate_loss(args, global_step, kl_all, alpha_i, recon_loss, model)
            if args.model_ring_artifact:
                loss_ring, _, _, _, _, _, _ = calculate_loss(args, global_step, kl_all_ring, alpha_i, recon_loss, model_ring)

        total_loss = loss + loss_ring if args.model_ring_artifact else loss
        grad_scalar.scale(total_loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        if args.model_ring_artifact:
            utils.average_gradients(model_ring.parameters(), args.distributed)
            
        grad_scalar.step(cnn_optimizer)
        if args.model_ring_artifact:
            grad_scalar.step(cnn_optimizer_ring)

        grad_scalar.update()
        nelbo.update(loss.data, 1)
        if (global_step + 1) % int(args.save_interval//4) == 0:
            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)
            writer.add_scalar('train/recon_iter', torch.mean(utils.reconstruction_loss(sino_raw_dist, sparse_sinogram_raw, args.dataset, crop=model.crop_output)), global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_coeff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        if ((global_step + 1) % args.save_interval == 0) and (args.global_rank == 0):  # save only on 1 rank
            n = int(np.floor(np.sqrt(x.size(0))))

            x_img = x[:n*n]
            x_tiled = utils.tile_image(x_img, n)

            writer.add_image('input image', x_tiled, global_step)
            plt.figure()
            plt.imshow(x_tiled[0].detach().cpu().numpy())
            save_filepath = args.save + '/input_image_rank_' + str(args.global_rank) + '_' + str(global_step)+'.png'
            print('saving image: ' + save_filepath)
            plt.savefig(save_filepath)

            ground_truth = ground_truth[:n*n,None]
            ground_truth_tiled = utils.tile_image(ground_truth, n)
            writer.add_image('ground truth', ground_truth_tiled, global_step)
            plt.figure()
            plt.imshow(ground_truth_tiled[0].detach().cpu().numpy())
            plt.colorbar()
            save_filepath = args.save + '/ground_truth_rank_' + str(args.global_rank) + '_'  + str(global_step)+'.png'
            print('saving image: ' + save_filepath)
            plt.savefig(save_filepath)

            output_sinogram_raw = sino_raw_dist.mean if isinstance(sino_raw_dist, torch.distributions.bernoulli.Bernoulli) else sino_raw_dist.sample()
            output_sinogram_raw = output_sinogram_raw[:n*n]
            output_sinogram_raw = output_sinogram_raw[:,None,:,:]
            output_sinogram_raw = utils.tile_image(output_sinogram_raw, n)  

            sparse_sinogram_raw = sparse_sinogram_raw[:n*n]
            sparse_sinogram_raw = sparse_sinogram_raw[:,None,:,:]
            sparse_sinogram_tiled = utils.tile_image(sparse_sinogram_raw, n)  
            in_out_tiled = torch.cat((sparse_sinogram_tiled, output_sinogram_raw), dim=2)
            writer.add_image('sinogram reconstruction', in_out_tiled, global_step)
            plt.figure()
            plt.imshow(in_out_tiled[0].detach().cpu().numpy())
            save_filepath = args.save + '/sinogram_reconstruction_rank_' + str(args.global_rank) + '_'  + str(global_step)+'.png'
            print('saving image: ' + save_filepath)
            plt.savefig(save_filepath)

            phantom_sample = phantom
            phantom_sample = phantom_sample[:n*n]
            phantom_sample = torch.transpose(phantom_sample,2,3)
            phantom_sample = torch.transpose(phantom_sample,1,2)
            phantom_tiled = utils.tile_image(phantom_sample, n)

            writer.add_image('phantom_reconstruction', phantom_tiled, global_step)
            plt.figure()
            plt.imshow(phantom_tiled[0].detach().cpu().numpy())
            save_filepath = args.save + '/phantom_reconstruction_rank_' + str(args.global_rank) + '_' + str(global_step)+'.png'
            print('saving image: ' + save_filepath)
            plt.savefig(save_filepath)
            plt.close('all')

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test(valid_queue, model, model_ring, epoch, num_samples, args, logging, dataset_type='', rank=None, max_num_examples=2):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.train() # need to set to train to get consistent results

    h5_filename = args.save + '/eval_dataset_' + dataset_type + '_epoch_' + str(epoch) + '_rank_' + str(rank) + '.h5'
    print(h5_filename)
    num_examples = 0
    with h5py.File(h5_filename, 'w') as h5_file:
        for step, x_full in enumerate(valid_queue):
            print('Testing at: ' + str(step))
            x, sparse_sinogram_raw, sparse_sinogram, ground_truth, theta, x_size, object_id = parse_x_full(x_full, args)

            with torch.no_grad():
                nelbo, log_iw = [], []
                all_phantoms = []
                for k in range(num_samples):
                    logits, log_q, log_p, kl_all, kl_diag, \
                    logits_ring, log_q_ring, log_p_ring, kl_all_ring, kl_diag_ring, \
                    sino_raw_dist, phantom, recon_loss = \
                        process_decoder_output(x, args, model, model_ring, theta, sparse_sinogram_raw, x_size, object_id)
                    all_phantoms.append(phantom.cpu().numpy())
                    balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                    nelbo_batch = recon_loss + balanced_kl
                    
                    nelbo.append(nelbo_batch)
                    log_iw.append(utils.log_iw(sino_raw_dist, sparse_sinogram_raw, log_q, log_p, args.dataset, crop=model.crop_output))

                all_phantoms = np.concatenate(all_phantoms, axis=-1)
                example_group = h5_file.create_group(f'example_{step}')
                example_group.create_dataset('phantom', data=all_phantoms)
                example_group.create_dataset('sparse_sinogram', data=sparse_sinogram.cpu().numpy())
                example_group.create_dataset('ground_truth', data=ground_truth.cpu().numpy())
                example_group.create_dataset('theta', data=theta.cpu().numpy())
                example_group.create_dataset('init_reconstruction', data=x.cpu().numpy()[:,0,:,:][:,np.newaxis,:,:])
                

                nelbo = torch.mean(torch.stack(nelbo, dim=1))
                log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))
                num_examples += 1
  
            nelbo_avg.update(nelbo.data, x.size(0))
            neg_log_p_avg.update(- log_p.data, x.size(0))
            if num_examples >= max_num_examples:
                break
        
        utils.average_tensor(nelbo_avg.avg, args.distributed)
        utils.average_tensor(neg_log_p_avg.avg, args.distributed)

    print('test num examples is ' + str(num_examples))
    
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg

def _get_sync_file():
    """Logic for naming sync file using slurm env variables"""
    sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync.%s.%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    return sync_file

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    if args.use_nersc:
        os.environ['MASTER_PORT'] = '29500'
    else:
        os.environ['MASTER_ADDR'] = args.master_address
        os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)

    sync_file = _get_sync_file()
    dist.init_process_group(backend='nccl', init_method=sync_file, rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='training iterations between saving images of results')
    parser.add_argument('--log_wandb', action='store_true', default=False,
                        help='If this flag is passed, log results to wandb')
    # data
    parser.add_argument('--dataset', type=str, default='foam',
                        help='dataset type to use, dataset should be in format dataset_type')
    parser.add_argument('--truncate', type=int, default=None,
                        help='if not None, truncate the training dataset to this many examples')
    parser.add_argument('--use_h5', dest='use_h5', type=lambda x: x.lower() == 'true',
                    help='If True, load relevant data from h5 file at every iteration',default=False)
    parser.add_argument('--use_masks', dest='use_masks', type=lambda x: x.lower() == 'true',
                help='If True, use image of the masks stacked with the image reconstruction as input to the encoder',default=False)
    parser.add_argument('--final_train', type=lambda x: x.lower() == 'true', default=False,
            help='This flag is for the final evaluation of the train examples.')
    parser.add_argument('--final_valid', type=lambda x: x.lower() == 'true', default=False,
            help='This flag is for the final evaluation of the validation examples.')
    parser.add_argument('--final_test', type=lambda x: x.lower() == 'true', default=False,
                help='This flag is for the final evaluation of the test examples. This should only be run once for the final results.')
    # optimization
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', type=lambda x: x.lower() == 'true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=10,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    parser.add_argument('--temp', dest='temp_bernoulli', type=float, default=2.2,
                        help='temperature of relaxed bernoulli')
    # physics parameters
    parser.add_argument('--pnm', dest='pnm', type=float, default=1e3,
                        help='poisson noise multiplier, higher value means higher SNR')
    parser.add_argument('--pnm_start', dest='pnm_start', type=float, default=1e1,
                        help='starting value for poisson noise multiplier')
    parser.add_argument('--pnm_warmup_epochs', dest='pnm_warmup_epochs', type=float, default=10000,
                        help='number of epochs before pnm reaches pnm_fraction of the final value')
    parser.add_argument('--pnm_fraction', dest='pnm_fraction', type=float, default=0.9,
                        help='we reach this fraction of the final pnm value at the end of pnm_warmup_epochs')
    parser.add_argument('--model_ring_artifact', dest='model_ring_artifact', type=lambda x: x.lower() == 'true',
                    help='If True, attempt to correct for a ring artifact', default=False)
    
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    # parser.add_argument('--node_rank', type=int, default=0,
    #                     help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--use_nersc', action='store_true', default=False,
                        help='This flag is for running on NERSC.')

    
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save

    args.node_rank = int(os.environ['SLURM_PROCID'])
    print('node rank is: ' + str(args.node_rank))
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if args.log_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="CT_NVAE",
            name=args.save,
            # track hyperparameters and run metadata
            config=args
        )


    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)


