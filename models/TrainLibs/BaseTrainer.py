import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from glob import glob
from tqdm import tqdm
from os.path import join, exists, dirname, basename
import re
import os
import time


class BaseTrainer(object):
    def __init__(self, model, device, epochs, train_loader, val_loader, lr, optimizer_mode, scheduler_mode, loss_mode,
                 root_path, exp_name, scheduler_kwargs, alpha=1, exp_path=None, checkpoint_path=None, writer_path=None):
        # initial param
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.kwargs = scheduler_kwargs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_mode = optimizer_mode
        self.scheduler_mode = scheduler_mode
        self.loss_mode = loss_mode
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.losser = None
        self.scaler = GradScaler()
        self.current_epoch = 0
        self.alpha = alpha
        self.root_path = root_path
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        # create param
        self.exp_path = exp_path
        self.checkpoint_path = checkpoint_path
        self.writer_path = writer_path
        if self.exp_path is None or self.checkpoint_path is None or self.writer_path is None:
            self.create_experiment(root_path=root_path, exp_name=exp_name)
        self.losser = self.create_losser()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.writer = SummaryWriter(str(self.writer_path))

    def create_optimizer(self):
        if self.optimizer_mode.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_mode.lower() == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters())
        elif self.optimizer_mode.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)
        else:
            raise ValueError(f"Optimizer type error: {self.optimizer_mode}")
        return optimizer

    def create_scheduler(self):
        if self.scheduler_mode is None:
            return None

        if self.scheduler_mode.lower() == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=float(self.kwargs.get("max_lr", self.lr)),
                total_steps=int(self.kwargs.get("total_step", self.epochs * len(self.train_loader))),
                pct_start=float(self.kwargs.get("pct_start", 0.3)),
                div_factor=int(self.kwargs.get("div_factor", 25)),
                final_div_factor=float(self.kwargs.get("final_div_factor", 1e4)),
                verbose=bool(self.kwargs.get("verbose", True))
            )
        elif self.scheduler_mode.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=int(self.kwargs.get("step_size", 30)),
                gamma=float(self.kwargs.get("gamma", 0.1))
            )
        elif self.scheduler_mode.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=int(self.kwargs.get("T_max", self.epochs)),
                eta_min=float(self.kwargs.get("eta_min", 1e-6))
            )
        elif self.scheduler_mode.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode=str(self.kwargs.get("mode", "min")),
                factor=float(self.kwargs.get("factor", 0.1)),
                patience=int(self.kwargs.get("patience", 10)),
                verbose=bool(self.kwargs.get("verbose", True))
            )
        elif self.scheduler_mode.lower() == "exp":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=float(self.kwargs.get("gamma", 0.95))
            )
        else:
            raise ValueError(
                f"Unknown scheduler pattern: {self.scheduler_mode}, option: 'onecycle', 'step', 'cosine', 'plateau', 'exp'")
        return scheduler

    def create_experiment(self, root_path, exp_name):
        os.makedirs(str(root_path), exist_ok=True)
        self.exp_path = join(root_path, "experiments", exp_name)  # experiment path
        os.makedirs(self.exp_path, exist_ok=True)
        self.checkpoint_path = join(self.exp_path, "checkpoints")
        self.writer_path = join(self.exp_path, "summary")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.writer_path, exist_ok=True)

    def create_losser(self):
        if self.loss_mode == "cross_entropy":
            return F.cross_entropy
        elif self.loss_mode == "mse":
            return F.mse_loss
        elif self.loss_mode == "l1":
            return F.l1_loss
        elif self.loss_mode == "binary":
            return F.binary_cross_entropy
        elif self.loss_mode == "bce_logits":
            return nn.BCEWithLogitsLoss
        elif self.loss_mode is None or self.loss_mode == "binary_cross_entropy_with_logits":
            return F.binary_cross_entropy_with_logits
        else:
            return F.binary_cross_entropy_with_logits

    def train(self, train_step=None, val_step=None):
        if train_step is None or val_step is None:
            raise ValueError("The train_step and val_step methods must be specified.")

        start_epoch = self.load_least_checkpoint()
        try:
            for epoch in range(start_epoch, self.epochs):
                avg_train_loss, avg_val_loss = self.one_epoch(
                    epoch=epoch,
                    train_step=train_step,
                    val_step=val_step
                )
                # print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ")
        except KeyboardInterrupt:
            self.emergency_save(epoch, avg_val_loss)
        finally:
            self.writer.close()

    def one_epoch(self, epoch, train_step=None, val_step=None):
        if train_step is None or val_step is None:
            raise ValueError("Training/validation step method not specified")

        '''
        loss_batch: 当前批次的平均损失值
        '''

        # train phase
        start_train_time = time.time()
        train_loss = 0.0
        self.model.train()
        with tqdm(self.train_loader, desc=f"Training epoch {epoch}", postfix={'loss': '?.0000'}) as pbar:
            for batch_idx, batch in enumerate(pbar):
                loss_batch, other_msg = train_step(batch)
                train_loss += loss_batch
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_batch:.4f}",
                        'avg': f"{(train_loss / (batch_idx + 1)):.4f}",
                        'msg': other_msg,
                        'time': f"{round((time.time() - start_train_time) / 60, 2):.2f}m"
                    })
        avg_train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0  # average loss

        # val phase
        start_val_time = time.time()
        val_loss = 0.0
        self.model.eval()
        with tqdm(self.val_loader, desc=f"Val Epoch {epoch}", postfix={'loss': '?.0000'}) as pbar:
            for batch_idx, batch in enumerate(pbar):
                loss_batch, other_msg = val_step(batch)
                val_loss += loss_batch
                pbar.set_postfix({
                    'loss': f"{loss_batch:.4f}",
                    'avg': f"{(val_loss / (batch_idx + 1)):.4f}",
                    'msg': other_msg,
                    'time': f"{round((time.time() - start_val_time) / 60, 2):.2f}m"
                })
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0  # average loss

        # update scheduler
        if self.scheduler_mode is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_val_loss)
            else:
                self.scheduler.step()

        # record
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log_metrics(epoch, avg_train_loss, avg_val_loss, current_lr)

        # save checkpoint
        self.save_checkpoint(epoch, avg_val_loss, self.alpha)

        return avg_train_loss, avg_val_loss

    def train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        with autocast():
            outputs = self.model(inputs)
            loss = self.losser(outputs, targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item(), None

    def val_step(self, batch):
        self.model.eval()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad(), autocast():
            outputs = self.model(inputs)
            loss = self.losser(outputs, targets)
        return loss.item(), None

    def save_checkpoint(self, epoch, loss, alpha=None):
        checkpoint_name = f"checkpoint_epoch{epoch}_loss{loss:.4f}.pt"
        checkpoint_file = os.path.join(self.checkpoint_path, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'alpha': alpha if alpha or alpha is not None else 0.0,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler_state_dict': self.scaler.state_dict(),
        }, checkpoint_file)

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_epoch = epoch
            best_file = os.path.join(self.checkpoint_path, "best_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'loss': loss,
                'alpha': alpha if alpha or alpha is not None else 0.0,
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch
            }, best_file)
        # print(f"Checkpoint saved to {checkpoint_file}")

    def load_least_checkpoint(self):
        checkpoints = glob(os.path.join(self.checkpoint_path, "checkpoint_epoch*.pt"))
        if not checkpoints:
            return 0  # no checkpoint

        def extract_epoch(fn):
            match = re.search(r'checkpoint_epoch(\d+)_', fn)
            return int(match.group(1)) if match else -1

        checkpoints.sort(key=extract_epoch)
        latest_checkpoint = checkpoints[-1]

        # print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.alpha = checkpoint.get('alpha', 2000)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        return checkpoint['epoch'] + 1

    def load_best_checkpoint(self):
        best_list = glob(os.path.join(self.checkpoint_path, "best_checkpoint.pt"))
        if not best_list:
            raise FileNotFoundError("No best checkpoint available")
        best_list.sort(key=os.path.getmtime)
        best_file = best_list[-1]

        checkpoint = torch.load(best_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'best_epoch': checkpoint['best_epoch']
        }

    def log_metrics(self, epoch, train_loss, val_loss, lr):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Learning Rate', lr, epoch)
        self.writer.add_scalar('Param/alpha', self.alpha, epoch)
        if epoch == self.best_epoch:
            self.writer.add_text('Best', '★', epoch)
        self.writer.add_scalar('Best/loss', self.best_val_loss, epoch)

    def emergency_save(self, epoch, loss):
        checkpoint_file = os.path.join(self.checkpoint_path, "emergency_save.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'loss': loss,
            'scaler_state_dict': self.scaler.state_dict()
        }, checkpoint_file)
        # print(f"Emergency checkpoint saved to {checkpoint_file}")
