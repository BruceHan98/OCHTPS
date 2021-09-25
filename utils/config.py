import yaml


class Config:
    def __init__(self, path):
        self.config_path = path
        self._get_config()

    def _get_config(self):
        with open(self.config_path, "r") as setting:
            config = yaml.load(setting)
        self.train_data_dir = config['train_data_dir']
        self.eval_data_dir = config['eval_data_dir']
        self.train_batch_size = config['train_batch_size']
        self.eval_batch_size = config['eval_batch_size']
        self.num_workers = config['num_workers']
        self.num_classes = config['num_classes']
        self.line_height = config['line_height']
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.display_interval = config['display_interval']
        self.print_pred = config['print_pred']

    def list_all_member(self):
        for name, value in vars(self).items():
            print('%s: %s' % (name, value))
