# FILE: model_interface_base.py

import inspect
import importlib
import pytorch_lightning as pl
from MyOptimizer import create_optimizer


class ModelInterfaceBase(pl.LightningModule):
    """
    模型接口的基类.

    这个类包含了所有任务（如分类、生存分析）共享的通用逻辑:
    1. 动态模型加载 (load_model, instancialize)
    2. 优化器配置 (configure_optimizers)
    3. 通用初始化逻辑
    4. Pytorch Lightning 的一些通用设置 (get_progress_bar_dict)
    """

    def __init__(self, model, optimizer, **kargs):
        """
        基类的构造函数.

        Args:
            model (object): 包含模型名称和参数的配置对象.
            optimizer (object): 包含优化器类型和参数的配置对象.
            **kargs: 包含日志路径等其他参数的字典.
        """
        super().__init__()
        # 使用 save_hyperparameters 保存核心配置，以便 Pytorch Lightning 跟踪
        # 注意：我们只保存对所有子类都通用的参数
        self.save_hyperparameters('model', 'optimizer')

        # 将一些常用配置保存为实例属性以便访问
        self.log_path = kargs.get('log')
        self.n_classes = model.n_classes

        # 将优化器配置保存起来，供 configure_optimizers 方法使用
        self.optimizer_cfg = optimizer

        # 动态加载并实例化模型
        # self.model 会在 load_model() 中被赋值
        self.model = None
        self.load_model()

    def load_model(self):
        """
        根据配置文件动态地从 'models' 文件夹加载模型.
        它会将 'model_name.py' 这样的文件名转换为 'ModelName' 这样的类名.
        """
        name = self.hparams.model.name
        # 将下划线命名（snake_case）转换为驼峰命名（CamelCase）
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name

        try:
            # 动态导入模块并获取模型类
            Model = getattr(importlib.import_module(f'models.{name}'), camel_name)
        except Exception as e:
            raise ValueError(f"无法加载模型。请检查 'models/{name}.py' 文件和 '{camel_name}' 类是否存在。\n错误: {e}")

        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """
        使用 hparams 中的参数实例化模型.
        """
        # 获取模型类的 __init__ 方法需要的参数
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()

        args_to_pass = {}
        for arg in class_args:
            if arg in inkeys:
                args_to_pass[arg] = getattr(self.hparams.model, arg)

        # 允许传入额外的参数来覆盖 hparams 中的值
        args_to_pass.update(other_args)
        return Model(**args_to_pass)

    def configure_optimizers(self):
        """
        使用工厂函数创建优化器.
        """
        optimizer = create_optimizer(self.optimizer_cfg, self.model)
        return [optimizer]

    def get_progress_bar_dict(self):
        """
        重写此方法以从进度条中移除 'v_num'.
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items