"""
模型配置文件
定义不同数据集的配置参数
"""

# 数据集配置
DATASET_CONFIG = {
    'RML2016.10a': {
        'num_classes': 11,
        'signal_length': 128,
        'classes': ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    },
    'RML2016.10b': {
        'num_classes': 11,
        'signal_length': 128,
        'classes': ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    },
    'RML2018a': {
        'num_classes': 24,
        'signal_length': 1024,
        'classes': ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                   '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                   '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                   'FM', 'GMSK', 'OQPSK']
    },
    'HisarMod': {
        'num_classes': 26,
        'signal_length': 1024,
        'classes': ['2FSK', '2PSK', '4FSK', '4PSK', '8FSK', '8PSK', '16PSK', '32PSK',
                   '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-DSB', 'AM-SSB',
                   'BPSK', 'CPFSK', 'GFSK', 'MSK', 'OQPSK', 'PAM4', 'QAM16', 'QAM64',
                   'QPSK', 'WBFM', 'GMSK']
    }
}


def get_dataset_config(dataset_name):
    """
    获取数据集配置
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        配置字典
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"未知数据集: {dataset_name}. 支持的数据集: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]

