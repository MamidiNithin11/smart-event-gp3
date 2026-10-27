from importlib import import_module

__all__ = [
    'finish_agent_run',
    'get_recent_runs',
    'get_run_tool_calls',
    'log_tool_call',
    'start_agent_run',
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(import_module('.logger', __name__), name)