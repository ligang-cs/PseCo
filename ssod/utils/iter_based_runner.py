from mmcv.runner import RUNNERS
from mmcv.runner.iter_based_runner import IterBasedRunner

@RUNNERS.register_module()
class IterBasedRunner_custom(IterBasedRunner):
    """ Pass iteration information to the model. 
    """
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, self._inner_iter, **kwargs)
        
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1