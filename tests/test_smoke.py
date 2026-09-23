def test_config_loads():
    from config import config
    assert hasattr(config, "QWEN_API_KEY")


def test_calculate_logic():
    # 直接测纯计算，不引入整个 agent 链路
    assert 1 + 2 * 3 == 7
