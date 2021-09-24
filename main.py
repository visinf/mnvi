from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import subprocess

import torch

import commandline
import configuration as config
import logger
import runtime
from utils import zipsource


def main():
    # ----------------------------------------------------
    # Change working directory
    # ----------------------------------------------------
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # ----------------------------------------------------
    # Parse commandline arguments
    # ----------------------------------------------------
    args = commandline.setup_logging_and_parse_arguments(
        blocktitle="Commandline Arguments")

    with logger.LoggingBlock("Source Code", emph=True):
        # ----------------------------------------------------
        # Also archieve source code
        # ----------------------------------------------------
        dst = os.path.join(args.save, "src.zip")
        zipsource.create_zip(
            filename=os.path.join(args.save, "src.zip"),
            directory=os.getcwd())
        logging.info("Archieved code: %s" % dst)

    # ----------------------------------------------------
    # Set random seed, possibly on Cuda
    # ----------------------------------------------------
    config.configure_random_seed(args)

    # ------------------------------------------------------
    # Fetch data loaders. Quit if no data loader is present
    # ------------------------------------------------------
    train_loader, validation_loader = config.configure_data_loaders(args)

    # -------------------------------------------------------------------------
    # Check whether any dataset could be found
    # -------------------------------------------------------------------------
    success = any(loader is not None for loader in [train_loader, validation_loader])
    if not success:
        logging.info("No dataset could be loaded successfully. Please check dataset paths!")
        quit()

    # ----------------------------------------------------------
    # Configure model and loss.
    # ----------------------------------------------------------
    model_and_loss = config.configure_model_and_loss(args)

    # -----------------------------------------------------------
    # Cuda
    # -----------------------------------------------------------
    with logger.LoggingBlock("Device", emph=True):
        logging.info(args.device)
        if 'parallel' in args.device:
            device = 'parallel'
        else:
            device = torch.device(args.device)

    # --------------------------------------------------------
    # Print model visualization
    # --------------------------------------------------------
    if args.logging_model_graph:
        with logger.LoggingBlock("Model Graph", emph=True):
            logger.log_module_info(model_and_loss.model)
    if args.logging_loss_graph:
        with logger.LoggingBlock("Loss Graph", emph=True):
            logger.log_module_info(model_and_loss.loss)

    # -------------------------------------------------------------------------
    # Possibly resume from checkpoint
    # -------------------------------------------------------------------------
    checkpoint_saver, checkpoint_stats = config.configure_checkpoint_saver(args, model_and_loss)
    if checkpoint_stats is not None:
        logging.info("  Checkpoint Statistics:")
        for key, value in checkpoint_stats.items():
            logging.info("    {}: {}".format(key, value))

        # ---------------------------------------------------------------------
        # Set checkpoint stats
        # ---------------------------------------------------------------------
        if args.checkpoint_mode in ["resume_from_best", "resume_from_latest"]:
            args.start_epoch = checkpoint_stats["epoch"]

    # ---------------------------------------------------------------------
    # Checkpoint and save directory
    # ---------------------------------------------------------------------
    with logger.LoggingBlock("Save Directory", emph=True):
        logging.info("Save directory: %s" % args.save)
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # ----------------------------------------------------------
    # Configure optimizer
    # ----------------------------------------------------------
    optimizer = config.configure_optimizer(args, model_and_loss)

    # ----------------------------------------------------------
    # Configure learning rate
    # ----------------------------------------------------------
    lr_scheduler = config.configure_lr_scheduler(args, optimizer)

    # ------------------------------------------------------------
    # If this is just an evaluation: overwrite savers and epochs
    # ------------------------------------------------------------
    if args.evaluation:
        args.start_epoch = 1
        args.total_epochs = 1
        train_loader = None
        checkpoint_saver = None
        optimizer = None
        lr_scheduler = None

    # ----------------------------------------------------------
    # Cuda optimization
    # ----------------------------------------------------------
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True

    # ----------------------------------------------------------
    # Kickoff training, validation and/or testing
    # ----------------------------------------------------------
    torch.autograd.set_detect_anomaly(True)
    return runtime.exec_runtime(
        args,
        device=device,
        checkpoint_saver=checkpoint_saver,
        model_and_loss=model_and_loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader)


if __name__ == "__main__":
    main()
