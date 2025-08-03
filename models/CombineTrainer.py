import torch
import time
import sys
from torch.nn import functional as F
from tqdm import tqdm
from os.path import dirname

sys.path.append(dirname(__file__))
from TrainLibs.BaseTrainer import BaseTrainer


class CombineTrainer(BaseTrainer):
    def __init__(self, model, model_type, device, epochs, train_loader, val_loader, lr=1e-3, exp_name=None,
                 optimizer='Adam', scheduler=None, loss=None, root=None, kwargs=None, alpha=1, exp_path=None,
                 checkpoint_path=None, writer_path=None, model_attribute='PT'):
        super(CombineTrainer, self).__init__(model=model,
                                             device=device,
                                             epochs=epochs,
                                             train_loader=train_loader,
                                             val_loader=val_loader,
                                             lr=lr,
                                             optimizer_mode=optimizer,
                                             scheduler_mode=scheduler,
                                             loss_mode=loss,
                                             root_path=root,
                                             exp_name=exp_name,
                                             scheduler_kwargs=kwargs,
                                             alpha=alpha,
                                             exp_path=exp_path,
                                             checkpoint_path=checkpoint_path,
                                             writer_path=writer_path)
        self.model_type = model_type
        self.model_attribute = model_attribute

    def combine_train(self):
        if self._train_step is None or self._val_step is None:
            raise ValueError("The train_step and val_step methods must be specified.")

        start_epoch = self.load_least_checkpoint()
        try:
            for epoch in range(start_epoch, self.epochs):
                avg_train_loss, avg_val_loss = self._one_epoch(epoch=epoch)
        except KeyboardInterrupt:
            self.emergency_save(epoch, avg_val_loss)
        finally:
            self.writer.close()

    def _one_epoch(self, epoch):
        # train phase
        start_train_time = time.time()
        train_loss = 0.0
        self.model.train()
        with tqdm(self.train_loader, desc=f"Training epoch {epoch}", postfix={'loss': '?.0000'}) as pbar:
            for batch_idx, batch in enumerate(pbar):
                loss_batch, other_msg = self._train_step(batch)
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
                loss_batch, other_msg = self._val_step(batch)
                val_loss += loss_batch
                pbar.set_postfix({
                    'loss': f"{loss_batch:.4f}",
                    'avg': f"{(val_loss / (batch_idx + 1)):.4f}",
                    'msg': other_msg,
                    'time': f"{round((time.time() - start_val_time) / 60, 2):.2f}m"
                })
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0  # average loss

        # record
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log_metrics(epoch, avg_train_loss, avg_val_loss, current_lr)

        # save checkpoint
        self.save_checkpoint(epoch, avg_val_loss, self.alpha)

        return avg_train_loss, avg_val_loss

    def _load_data(self, batch):
        if self.model_type == 'PoseCorr':
            scan_points = batch.get('scan_points').float().to(self.device)
            correspondences = batch.get('correspondences').float().to(self.device)
            parts = batch.get('parts').long().to(self.device)
            inputs = {
                'input_points': scan_points,  # Any posture point cloud
            }
            outputs = {
                'parts': parts,  # Part
                'logics': correspondences  # T-pose correspondence
            }
        elif self.model_type == 'BackGeo':
            correspondences = batch.get('correspondences').float().to(self.device)
            correspondences_back = batch.get('correspondences_back').float().to(self.device)
            parts = batch.get('parts').long().to(self.device)
            inputs = {
                'input_points': correspondences,  # T-pose front correspondence
            }
            outputs = {
                'parts': parts,  # Part
                'logics': correspondences_back - correspondences  # T-pose back correspondence
            }
        else:
            raise 'The type of the model is not legal'
        return inputs, outputs

    def _train_step(self, batch):
        self.model.train()
        inputs, outputs = self._load_data(batch)
        self.optimizer.zero_grad()
        pred_outputs = self.model(**inputs)
        loss, parts_loss, logics_loss = self._compute_loss(pred_outputs, outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item(), f"parts_loss: {parts_loss.item():.8f}, logics_loss: {logics_loss.item():.8f}"

    def _val_step(self, batch):
        self.model.eval()
        inputs, outputs = self._load_data(batch)
        with torch.no_grad():
            pred_outputs = self.model(**inputs)
            loss, parts_loss, logics_loss = self._compute_loss(pred_outputs, outputs)
        return loss.item(), f"parts_loss: {parts_loss.item():.8f}, logics_loss: {logics_loss.item():.8f}"

    def _compute_loss(self, pred_outputs, target_outputs, inputs=None):
        if self.model_type == 'PoseCorr':
            pred_logics = pred_outputs['logics'].permute(0, 2, 1)  # torch.size([1, 10000, 3])
            pred_parts = pred_outputs['parts']  # torch.size([1, 14, 10000)]
            target_logics = target_outputs['logics']  # torch.size([1, 10000, 3])
            target_parts = target_outputs['parts']
            parts_loss = F.cross_entropy(pred_parts, target_parts)
            logics_loss = F.mse_loss(pred_logics, target_logics)
            logics_loss *= 1000

        elif self.model_type == 'BackGeo':
            pred_logics = pred_outputs['logics'].permute(0, 2, 1)  # torch.Size([10, 10000, 3])
            pred_parts = pred_outputs['parts']  # torch.Size([10, 14, 10000])
            target_logics = target_outputs['logics']  # torch.Size([10, 10000, 3])
            target_parts = target_outputs['parts']  # torch.Size([10, 10000])
            parts_loss = F.cross_entropy(pred_parts, target_parts)
            logics_loss = F.mse_loss(pred_logics[..., 2], target_logics[..., 2])
            logics_loss *= 1000
        else:
            raise 'The type of the model is not legal'
        return parts_loss + logics_loss, parts_loss, logics_loss
