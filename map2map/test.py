import os
import sys
import warnings
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.func import jvp

from .data import FieldDataset
from .data import norms
from . import models
from .models import narrow_cast
from .utils import import_attr, load_model_state_dict

def test(args):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            warnings.warn('Not parallelized but given more than 1 GPUs')

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda', 0)

        torch.backends.cudnn.benchmark = True
    else:  # CPU multithreading
        device = torch.device('cpu')

        if args.num_threads is None:
            args.num_threads = int(os.environ['SLURM_CPUS_ON_NODE'])

        torch.set_num_threads(args.num_threads)

    print('pytorch {}'.format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()

    test_dataset = FieldDataset(
        style_pattern=args.test_style_pattern,
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    style_size = test_dataset.style_size
    in_chan = test_dataset.in_chan
    out_chan = test_dataset.tgt_chan[:1]
    write_chan = (out_chan[0],) * 2

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(style_size, sum(in_chan), sum(out_chan),
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)

    criterion = import_attr(args.criterion, torch.nn, models,
                            callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):

            input = data['input'].to(device, non_blocking=True)
            Om = data['Om'].float().to(device)
            Dz = data['Dz'].float().to(device)

            model_eval = lambda tDz : model(input, Om, tDz)
            jvp_dir = torch.tensor([1.]).to(device)
            (out_dis, s), (out_vel, ds) = jvp(model_eval, (Dz,), (jvp_dir,))

            if i <= 5 :
                print('##### batch :', i)
                print('style :', s)
                print('style grad :', ds)
                print('input shape :', input.shape)
                print('out_dis shape :', out_dis.shape)
                print('out_vel shape :', out_vel.shape)
                print('write_chan :', write_chan)

            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip([test_dataset.tgt_norms[0]], np.cumsum(out_chan)):
                    norm = import_attr(norm, norms, callback_at=args.callback_at)
                    norm(out_dis[:, start:stop], undo=True, **args.misc_kwargs)
                    start = stop
                start = 0
                z = data['redshift']
                Om = data['Om']
                for norm, stop in zip([test_dataset.tgt_norms[1]], np.cumsum(out_chan)):
                    norm = import_attr(norm, norms, callback_at=args.callback_at)
                    x = torch.ones(1)
                    norm(x, undo=True, Om=Om, z=z, **args.misc_kwargs)
                    x = x.to(device)
                    out_vel *= x
                    start = stop

            output = torch.cat((out_dis, out_vel), 1)
            test_dataset.assemble('_out', write_chan, output, data['target_relpath'])
