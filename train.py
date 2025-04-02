import argparse
import datetime
import json
import os
import pandas as pd
import time
import torch
import torch.distributed as dist
from evaluate import SmilesEvaluator
from molscribe.chemistry import convert_graph_to_smiles, postprocess_smiles, keep_main_molecule
from molscribe.dataset import TrainDataset, polymer_collate
from molscribe.model import Encoder, Decoder
from molscribe.loss import Criterion
from molscribe.tokenizer import get_tokenizer
from molscribe.utils import seed_torch, save_args, init_summary_writer, LossMeter, AverageMeter, asMinutes, timeSince, \
    log_rank_0, format_df, init_logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
from typing import Any, Dict, Optional, Tuple


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_val', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'])
    # Model
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--decoder', type=str, default='lstm')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--enc_pos_emb', action='store_true')
    group = parser.add_argument_group("lstm_options")
    group.add_argument('--decoder_dim', type=int, default=512)
    group.add_argument('--decoder_layer', type=int, default=1)
    group.add_argument('--attention_dim', type=int, default=256)
    group = parser.add_argument_group("transformer_options")
    group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
    group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
    group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
    group.add_argument("--dec_num_queries", type=int, default=128)
    group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
    group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
    group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
    group.add_argument("--num_bond_type", help="Number of bond types including no bond", type=int, default=7)
    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_files', type=str, default=None)
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--test_files', type=str, default=None)
    parser.add_argument('--coords_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--default_option', action='store_true')
    parser.add_argument('--include_condensed', action='store_true')
    parser.add_argument('--formats', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--mol_augment', action='store_true')
    parser.add_argument('--coord_bins', type=int, default=100)
    parser.add_argument('--sep_xy', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0)
    parser.add_argument('--continuous_coords', action='store_true')
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--all_data', action='store_true', help='Use both train and valid data for training.')
    parser.add_argument('--init_scheduler', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--shuffle_nodes', action='store_true')
    parser.add_argument('--save_image', action='store_true')
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_best', type=int, default=1)
    parser.add_argument('--save_attns', action='store_true')
    parser.add_argument('--molblock', action='store_true')
    parser.add_argument('--compute_confidence', action='store_true')
    parser.add_argument('--keep_main_molecule', action='store_true')
    args = parser.parse_args()

    return args


def load_states(args, load_path):
    if load_path.endswith(".pth"):
        path = load_path
    elif args.load_ckpt == "best":
        path = os.path.join(load_path, f'{args.encoder}_{args.decoder}_best.pth')
    else:
        path = os.path.join(load_path, f'{args.encoder}_{args.decoder}_{args.load_ckpt}.pth')
    log_rank_0(f"Load {path}")
    states = torch.load(path, map_location=torch.device('cpu'))

    return states


def safe_load(module, module_states) -> None:
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = module.load_state_dict(
        remove_prefix(module_states),
        strict=False
    )
    if missing_keys:
        log_rank_0(f"Missing keys: {str(missing_keys)}")
    if unexpected_keys:
        log_rank_0(f"Unexpected keys: {str(unexpected_keys)}")


def safe_load_with_shape_change(module, module_states) -> None:
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    new_state_dict = module.state_dict()
    pretrained_state_dict = remove_prefix(module_states)

    # Extension for sequence decoder
    output_w = new_state_dict["decoder.chartok_coords.output_layer.weight"]
    output_b = new_state_dict["decoder.chartok_coords.output_layer.bias"]
    emb_luts = new_state_dict["decoder.chartok_coords.embeddings.make_embedding.emb_luts.0.weight"]

    pretrained_param = pretrained_state_dict["decoder.chartok_coords.output_layer.weight"]
    pretrained_dim = pretrained_param.size(0)
    # output_w[:, :] = 0
    output_w[:pretrained_dim] = pretrained_param

    pretrained_param = pretrained_state_dict["decoder.chartok_coords.output_layer.bias"]
    pretrained_dim = pretrained_param.size(0)
    # output_b[:] = 0
    output_b[:pretrained_dim] = pretrained_param
    # print(f"Pretrained output_b: {pretrained_param}, shape: {pretrained_param.size()}")
    # print(f"New output_b: {output_b}, shape: {output_b.size()}")

    pretrained_param = pretrained_state_dict["decoder.chartok_coords.embeddings.make_embedding.emb_luts.0.weight"]
    pretrained_dim = pretrained_param.size(0)
    # emb_luts[:, :] = 0
    emb_luts[:pretrained_dim] = pretrained_param

    new_state_dict = pretrained_state_dict
    new_state_dict["decoder.chartok_coords.output_layer.weight"] = output_w
    new_state_dict["decoder.chartok_coords.output_layer.bias"] = output_b
    new_state_dict["decoder.chartok_coords.embeddings.make_embedding.emb_luts.0.weight"] = emb_luts

    """
    # Extension for edge decoder (GraphPredictor)

    mlp_2_w = new_state_dict["decoder.edges.mlp.2.weight"]
    mlp_2_b = new_state_dict["decoder.edges.mlp.2.bias"]
    # print(f"Initial mlp_2_b: {mlp_2_b}")

    mlp_2_w[:7] = pretrained_state_dict["decoder.edges.mlp.2.weight"]
    mlp_2_b[:7] = pretrained_state_dict["decoder.edges.mlp.2.bias"]
    # print(f"mlp_2_b after loading pretrained: {mlp_2_b}")

    new_state_dict = pretrained_state_dict
    new_state_dict["decoder.edges.mlp.2.weight"] = mlp_2_w
    new_state_dict["decoder.edges.mlp.2.bias"] = mlp_2_b
    # print(f"mlp_2_b to be loaded into the new module: {mlp_2_b}")
    """

    module.load_state_dict(new_state_dict)


def get_model(args, tokenizer, device, load_path=None):
    encoder = Encoder(args, pretrained=(not args.no_pretrained and load_path is None))
    args.encoder_dim = encoder.n_features
    log_rank_0(f"encoder_dim: {args.encoder_dim}")

    decoder = Decoder(args, tokenizer)
    if load_path:
        states = load_states(args, load_path)
        safe_load(encoder, states["encoder"])
        # safe_load(decoder, states["decoder"])
        safe_load_with_shape_change(decoder, states["decoder"])

        log_rank_0(f"Model loaded from {load_path}")
    encoder.to(device)
    decoder.to(device)

    if args.local_rank != -1:
        encoder = DDP(encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        decoder = DDP(decoder, device_ids=[args.local_rank], output_device=args.local_rank)
        log_rank_0("DDP setup finished")

    return encoder, decoder


def get_optimizer_and_scheduler(args, encoder, decoder, load_path=None):
    encoder_optimizer = AdamW(
        encoder.parameters(),
        lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        amsgrad=False
    )
    encoder_scheduler = get_scheduler(
        args.scheduler,
        encoder_optimizer,
        args.num_warmup_steps,
        args.num_training_steps
    )

    decoder_optimizer = AdamW(
        decoder.parameters(),
        lr=args.decoder_lr,
        weight_decay=args.weight_decay,
        amsgrad=False
    )
    decoder_scheduler = get_scheduler(
        args.scheduler,
        decoder_optimizer,
        args.num_warmup_steps,
        args.num_training_steps
    )

    if load_path and args.resume:
        states = load_states(args, load_path)
        encoder_optimizer.load_state_dict(states['encoder_optimizer'])
        decoder_optimizer.load_state_dict(states['decoder_optimizer'])
        if args.init_scheduler:
            for group in encoder_optimizer.param_groups:
                group['lr'] = args.encoder_lr
            for group in decoder_optimizer.param_groups:
                group['lr'] = args.decoder_lr
        else:
            encoder_scheduler.load_state_dict(states['encoder_scheduler'])
            decoder_scheduler.load_state_dict(states['decoder_scheduler'])
        log_rank_0(f"Optimizer loaded from {load_path}")

    return encoder_optimizer, encoder_scheduler, decoder_optimizer, decoder_scheduler


def train_fn(
    args, train_loader, encoder, decoder, criterion,
    encoder_optimizer, decoder_optimizer, epoch,
    encoder_scheduler, decoder_scheduler,
    scaler, device, global_step
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = LossMeter()
    seq_acc_meter = LossMeter()
    seq_acc_token_only_meter = LossMeter()
    edge_tp_meter = LossMeter()

    # switch to train mode
    encoder.train()
    decoder.train()
    
    start = end = time.time()
    encoder_grad_norm = decoder_grad_norm = 0

    for step, (indices, images, refs) in enumerate(train_loader):
        # print(indices)
        # for k, v in refs.items():
        #     print(k)
        #     print(v)
        # exit(0)

        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            features, hiddens = encoder(images)
            results = decoder(encoder_out=features, refs=refs)
            losses, metrics = criterion(results, refs)

            # logits, target, _ = results["chartok_coords"]
            # print(torch.argmax(logits, dim=-1))
            # print(target)
            # exit(0)

            loss = sum(losses.values())
            seq_acc = metrics["seq_acc"]
            seq_acc_token_only = metrics["seq_acc_token_only"]
            edge_tp = metrics["edge_tp"]

        # record loss
        loss_meter.update(
            loss,
            losses,
            batch_size
        )
        seq_acc_meter.update(
            seq_acc,
            {k: v for k, v in metrics.items() if k == "seq_acc"},
            batch_size
        )
        seq_acc_token_only_meter.update(
            seq_acc_token_only,
            {k: v for k, v in metrics.items() if k == "seq_acc_token_only"},
            batch_size
        )
        edge_tp_meter.update(
            edge_tp,
            {k: v for k, v in metrics.items() if k == "edge_tp"},
            batch_size
        )

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(encoder_optimizer)
            scaler.unscale_(decoder_optimizer)
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
            scaler.step(encoder_optimizer)
            scaler.step(decoder_optimizer)
            scaler.update()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_scheduler.step()
            decoder_scheduler.step()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            loss_str = ' '.join([f'{k}:{v.avg:.4f}' for k, v in loss_meter.subs.items()])
            log_rank_0(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Data {data_time.avg:.3f}s ({asMinutes(data_time.sum)}) "
                f"Run {timeSince(start, float(step + 1) / len(train_loader)):s} "
                f"Loss: {loss_meter.avg:.4f} ({loss_str}) "
                f"Seq. acc.: {seq_acc_meter.avg:.4f} "
                f"Seq. acc. token only: {seq_acc_token_only_meter.avg:.4f} "
                f"Edge true pos: {edge_tp_meter.avg:.4f} "
                f"Grad: {encoder_grad_norm:.3f}/{decoder_grad_norm:.3f} "
                f"LR: {encoder_scheduler.get_lr()[0]:.6f} "
                f"{decoder_scheduler.get_lr()[0]:.6f}"
            )
            loss_meter.reset()
            seq_acc_meter.reset()
            seq_acc_token_only_meter.reset()
            edge_tp_meter.reset()
        if args.train_steps_per_epoch != -1 and \
            (step + 1) // args.gradient_accumulation_steps == args.train_steps_per_epoch:
            break

    return loss_meter.epoch.avg, global_step


def val_fn(
    args,
    val_loader,
    encoder,
    decoder,
    criterion,
    tokenizer,
    device
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = LossMeter()
    seq_acc_meter = LossMeter()
    seq_acc_token_only_meter = LossMeter()
    edge_tp_meter = LossMeter()

    # switch to evaluation mode
    if hasattr(decoder, 'module'):
        encoder = encoder.module
        decoder = decoder.module
    encoder.eval()
    decoder.eval()
    predictions = {}
    start = end = time.time()
    # Inference is no longer distributed.
    for step, (indices, images, refs) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                features, hiddens = encoder(images)
                results = decoder(encoder_out=features, refs=refs)
                losses, metrics = criterion(results, refs)

                # logits, target, _ = results["chartok_coords"]
                # print(torch.argmax(logits, dim=-1))
                # print(target)
                # exit(0)

                loss = sum(losses.values())
                seq_acc = metrics["seq_acc"]
                seq_acc_token_only = metrics["seq_acc_token_only"]
                edge_tp = metrics["edge_tp"]

                batch_preds = decoder.decode(encoder_out=features)

        for i, ind in enumerate(indices):
            if ind > 5:
                continue
            log_rank_0(f"Prediction for {ind}: ")
            log_rank_0(f"ref: {refs['chartok_coords'][0][i]}")
            for k, v in batch_preds[i].items():
                if k in ["edges", "edge_scores"]:
                    continue
                log_rank_0(f"{k}: {v}")

        # record loss
        loss_meter.update(
            loss,
            losses,
            batch_size
        )
        seq_acc_meter.update(
            seq_acc,
            {k: v for k, v in metrics.items() if k == "seq_acc"},
            batch_size
        )
        seq_acc_token_only_meter.update(
            seq_acc_token_only,
            {k: v for k, v in metrics.items() if k == "seq_acc_token_only"},
            batch_size
        )
        edge_tp_meter.update(
            edge_tp,
            {k: v for k, v in metrics.items() if k == "edge_tp"},
            batch_size
        )

        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps

        for idx, preds in zip(indices, batch_preds):
            predictions[idx] = preds
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(val_loader) - 1):
            loss_str = ' '.join([f'{k}:{v.avg:.4f}' for k, v in loss_meter.subs.items()])
            log_rank_0(
                f"EVAL: [{step}/{len(val_loader)}] "
                f"Data {data_time.avg:.3f}s ({asMinutes(data_time.sum)}) "
                f"Elapsed {timeSince(start, float(step + 1) / len(val_loader)):s} "
                f"Loss: {loss_meter.avg:.4f} ({loss_str}) "
                f"Seq. acc.: {seq_acc_meter.avg:.4f} "
                f"Seq. acc. token only: {seq_acc_token_only_meter.avg:.4f} "
                f"Edge true pos: {edge_tp_meter.avg:.4f} "
            )
    # # gather predictions from different GPUs
    # gathered_preds = [None for i in range(dist.get_world_size())]
    # dist.all_gather_object(gathered_preds, predictions)
    # n = len(valid_loader.dataset)
    # predictions = [{}] * n
    # for preds in gathered_preds:
    #     for idx, pred in preds.items():
    #         predictions[idx] = pred

    return predictions


def train_loop(
    args,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: Dict[str, Any],
    save_path: str
) -> None:
    SUMMARY = None

    if args.local_rank == 0 and not args.debug:
        os.makedirs(save_path, exist_ok=True)
        save_args(args)
        SUMMARY = init_summary_writer(save_path)

    log_rank_0("========== training ==========")

    device = args.device

    # ====================================================
    # loader
    # ====================================================

    train_dataset = TrainDataset(args, train_df, tokenizer, split="train")
    log_rank_0(train_dataset.transform)

    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=polymer_collate
    )

    if args.train_steps_per_epoch == -1:
        args.train_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    args.num_training_steps = args.epochs * args.train_steps_per_epoch
    args.num_warmup_steps = int(args.num_training_steps * args.warmup_ratio)

    # ====================================================
    # model & optimizer
    # ====================================================
    if args.resume and args.load_path is None:
        args.load_path = args.save_path
    encoder, decoder = get_model(args, tokenizer, device, load_path=args.load_path)
    encoder_optimizer, encoder_scheduler, decoder_optimizer, decoder_scheduler = \
        get_optimizer_and_scheduler(args, encoder, decoder, load_path=args.load_path)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ====================================================
    # loop
    # ====================================================
    criterion = Criterion(args, tokenizer).to(device)

    global_step = encoder_scheduler.last_epoch
    start_epoch = global_step // args.train_steps_per_epoch

    for epoch in range(start_epoch, args.epochs):

        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
            dist.barrier()

        start_time = time.time()

        # train
        avg_loss, global_step = train_fn(
            args=args,
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            encoder_scheduler=encoder_scheduler,
            decoder_scheduler=decoder_scheduler,
            scaler=scaler,
            device=device,
            global_step=global_step
        )

        # eval
        # if not epoch == 6:
        #     continue
        scores = inference(
            args=args,
            data_df=val_df,
            criterion=criterion,
            tokenizer=tokenizer,
            encoder=encoder,
            decoder=decoder,
            save_path=save_path,
            split="val"
        )

        if args.local_rank > 0:
            continue

        elapsed = time.time() - start_time

        log_rank_0(f"Epoch {epoch + 1} - Time: {elapsed:.0f}s")
        log_rank_0(f"Epoch {epoch + 1} - Score: {json.dumps(scores)}")

        """
        save_obj = {
            'encoder': encoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'encoder_scheduler': encoder_scheduler.state_dict(),
            'decoder': decoder.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'decoder_scheduler': decoder_scheduler.state_dict(),
            'global_step': global_step,
            'args': {
                key: args.__dict__[key]
                for key in ['formats', 'input_size', 'coord_bins', 'sep_xy']
            }
        }

        for name in ['post_smiles', 'graph_smiles', 'canon_smiles']:
            if name in scores:
                score = scores[name]
                break

        if SUMMARY:
            SUMMARY.add_scalar('train/loss', avg_loss, global_step)
            encoder_lr = encoder_scheduler.get_lr()[0]
            decoder_lr = decoder_scheduler.get_lr()[0]
            SUMMARY.add_scalar('train/encoder_lr', encoder_lr, global_step)
            SUMMARY.add_scalar('train/decoder_lr', decoder_lr, global_step)
            for key in scores:
                SUMMARY.add_scalar(f'valid/{key}', scores[key], global_step)

        if score >= best_score:
            best_score = score
            log_rank_0(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_best.pth'))
            with open(os.path.join(save_path, 'best_valid.json'), 'w') as f:
                json.dump(scores, f)
        
        

        if args.save_mode == "all":
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_ep{epoch}.pth'))
        if args.save_mode == "last":
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_last.pth'))
        
        """

    if args.local_rank != -1:
        dist.barrier()


def inference(
    args, data_df, criterion, tokenizer,
    encoder=None, decoder=None, save_path=None, split: str = "test"
):
    log_rank_0("========== inference ==========")
    log_rank_0(data_df.attrs['file'])

    if args.local_rank == 0 and os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    device = args.device

    dataset = TrainDataset(args, data_df, tokenizer, split=split)
    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size * 2,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=polymer_collate
    )
    if encoder is None or decoder is None:
        # valid/test mode
        if args.load_path is None:
            args.load_path = save_path
        encoder, decoder = get_model(args, tokenizer, device, args.load_path)

    predictions = val_fn(
        args=args,
        val_loader=dataloader,
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device
    )

    return
    # The evaluation and saving prediction is only performed in the master process.
    if args.local_rank > 0:
        return
    log_rank_0("Start evaluation")

    # Deal with discrepancies between datasets
    if 'image_id' not in data_df.columns:
        data_df['image_id'] = [path.split('/')[-1].split('.')[0] for path in data_df['file_path']]
    pred_df = data_df[['image_id']].copy()
    scores = {}

    for format_ in args.formats:
        if format_ in ['atomtok', 'atomtok_coords', 'chartok_coords']:
            format_preds = [preds[format_] for preds in predictions]
            # SMILES
            pred_df['SMILES'] = [preds['smiles'] for preds in format_preds]
            if format_ in ['atomtok_coords', 'chartok_coords']:
                pred_df['node_coords'] = [preds['coords'] for preds in format_preds]
                pred_df['node_symbols'] = [preds['symbols'] for preds in format_preds]
            if args.compute_confidence:
                pred_df['SMILES_scores'] = [preds['scores'] for preds in format_preds]
                pred_df['indices'] = [preds['indices'] for preds in format_preds]

    # Construct graph from predicted atoms and bonds (including verify chirality)
    if 'edges' in args.formats:
        pred_df['edges'] = [preds['edges'] for preds in predictions]
        if args.compute_confidence:
            pred_df['edges_scores'] = [preds['edges_scores'] for preds in predictions]
        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            pred_df['node_coords'], pred_df['node_symbols'], pred_df['edges'])

        print(f'Graph to SMILES success ratio: {r_success:.4f}')
        pred_df['graph_SMILES'] = smiles_list
        if args.molblock:
            pred_df['molblock'] = molblock_list

    # Postprocess the predicted SMILES (verify chirality, expand functional groups)
    if 'SMILES' in pred_df.columns:
        if 'edges' in pred_df.columns:
            smiles_list, _, r_success = postprocess_smiles(
                pred_df['SMILES'], pred_df['node_coords'], pred_df['node_symbols'], pred_df['edges'])
        else:
            smiles_list, _, r_success = postprocess_smiles(pred_df['SMILES'])
        print(f'Postprocess SMILES success ratio: {r_success:.4f}')
        pred_df['post_SMILES'] = smiles_list

    # Keep the main molecule
    if args.keep_main_molecule:
        if 'graph_SMILES' in pred_df:
            pred_df['graph_SMILES'] = keep_main_molecule(pred_df['graph_SMILES'])
        if 'post_SMILES' in pred_df:
            pred_df['post_SMILES'] = keep_main_molecule(pred_df['post_SMILES'])

    # Compute scores
    if 'SMILES' in data_df.columns:
        evaluator = SmilesEvaluator(data_df['SMILES'], tanimoto=True)
        print('label:', data_df['SMILES'].values[:2])
        if 'SMILES' in pred_df.columns:
            print('pred:', pred_df['SMILES'].values[:2])
            scores.update(evaluator.evaluate(pred_df['SMILES']))
        if 'post_SMILES' in pred_df.columns:
            post_scores = evaluator.evaluate(pred_df['post_SMILES'])
            scores['post_smiles'] = post_scores['canon_smiles']
            scores['post_graph'] = post_scores['graph']
            scores['post_chiral'] = post_scores['chiral']
            scores['post_tanimoto'] = post_scores['tanimoto']
        if 'graph_SMILES' in pred_df.columns:
            graph_scores = evaluator.evaluate(pred_df['graph_SMILES'])
            scores['graph_smiles'] = graph_scores['canon_smiles']
            scores['graph_graph'] = graph_scores['graph']
            scores['graph_chiral'] = graph_scores['chiral']
            scores['graph_tanimoto'] = graph_scores['tanimoto']

    print('Save predictions...')
    file = data_df.attrs['file'].split('/')[-1]
    pred_df = format_df(pred_df)
    pred_df.to_csv(os.path.join(save_path, f'prediction_{file}'), index=False)
    # Save scores
    if split == 'test':
        with open(os.path.join(save_path, f'eval_scores_{os.path.splitext(file)[0]}_{args.load_ckpt}.json'), 'w') as f:
            json.dump(scores, f)

    return scores


def get_data(args) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Dict[str, Any]
]:
    train_df, val_df, test_df = None, None, None
    if args.do_train:
        train_files = args.train_files.split(',')
        train_df = pd.concat([
            pd.read_csv(os.path.join(args.data_path, file))
            for file in train_files
        ])
        log_rank_0(f'train.shape: {train_df.shape}')
    if args.do_train or args.do_val:
        val_df = pd.read_csv(os.path.join(args.data_path, args.val_file))
        val_df.attrs['file'] = args.val_file
        log_rank_0(f'val.shape: {val_df.shape}')
    if args.do_test:
        test_files = args.test_files.split(',')
        test_df = [
            pd.read_csv(os.path.join(args.data_path, file))
            for file in test_files
        ]
        for file, df in zip(test_files, test_df):
            df.attrs['file'] = file
            log_rank_0(f'{file} test.shape: {df.shape}')
    tokenizer = get_tokenizer(args)

    return train_df, val_df, test_df, tokenizer


def main(args):
    seed_torch(seed=args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank != -1:
        dist.init_process_group(
            backend=args.backend,
            init_method='env://',
            timeout=datetime.timedelta(0, 14400)
        )
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = False

    args.formats = args.formats.split(',')
    args.nodes = True
    args.edges = True
    log_rank_0('Output formats: ' + ' '.join(args.formats))

    train_df, val_df, test_df, tokenizer = get_data(args)

    if args.do_train:
        train_loop(
            args,
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            save_path=args.save_path
        )

    if args.do_val:
        scores = inference(
            args,
            data_df=val_df,
            criterion=None,
            tokenizer=tokenizer,
            save_path=args.save_path,
            split="val"
        )
        log_rank_0(json.dumps(scores, indent=4))

    if args.do_test:
        assert type(test_df) is list
        for df in test_df:
            scores = inference(
                args,
                data_df=df,
                criterion=None,
                tokenizer=tokenizer,
                save_path=args.save_path,
                split="test"
            )
            log_rank_0(json.dumps(scores, indent=4))


if __name__ == "__main__":
    args = get_args()
    logger = init_logger(args)

    main(args)
