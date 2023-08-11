#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import TypeVar, Type, Dict, List
import logging

from toolbox.open_nmt.common.from_params import FromParams
from toolbox.open_nmt.common.testing.checks import ConfigurationError


logger = logging.getLogger(__name__)


T = TypeVar('T')


class Registrable(FromParams):
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None
    register_name: str = 'unknown'

    @classmethod
    def register(cls: Type[T], name: str, exist_ok=False):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            # set a name on the subclass
            setattr(subclass, 'register_name', name)
            if name in registry:
                if exist_ok:
                    message = (f"{name} has already been registered as {registry[name].__name__}, but "
                               f"exist_ok=True, so overwriting with {cls.__name__}")
                    # logger.info(message)
                else:
                    message = (f"Cannot register {name} as {cls.__name__}; "
                               f"name already in use for {registry[name].__name__}")
                    raise ValueError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        # logger.info(f"instantiating registered subclass {name} of {cls}")
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls].get(name)
        else:
            raise ValueError(
                f"{name} is not a registered name for {cls.__name__}. "
                f"the available is: [{Registrable._registry[cls].keys()}]"
            )

    @classmethod
    def list_available(cls) -> List[str]:
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            message = "Default implementation %s is not registered" % default
            raise ValueError(message)
        else:
            return [default] + [k for k in keys if k != default]
