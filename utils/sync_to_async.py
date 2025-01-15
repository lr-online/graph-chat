import asyncio
from functools import wraps
from typing import Callable


def sync_or_async(func: Callable):
    """装饰器：根据调用环境适配同步或异步方法"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            if asyncio.get_event_loop().is_running():
                # 如果事件循环正在运行，直接调异步方法
                return func(*args, **kwargs)
            else:
                # 否则，通过 asyncio.run 执行异步方法
                return asyncio.run(func(*args, **kwargs))
        else:
            # 如果是同步方法，直接调用
            return func(*args, **kwargs)

    return wrapper
