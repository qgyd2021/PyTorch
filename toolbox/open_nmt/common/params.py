#!/usr/bin/python3
# -*- coding: utf-8 -*-
import collections
import copy
import inspect
import json
import logging
from typing import TypeVar, Type, Dict, Union, Any, cast, List, Tuple, Set, MutableMapping, OrderedDict
import zlib

from overrides import overrides

from toolbox.open_nmt.common.testing.checks import ConfigurationError


logger = logging.getLogger(__name__)


T = TypeVar('T')


_NO_DEFAULT = inspect.Parameter.empty


def infer_and_cast(value: Any):
    """
    In some cases we'll be feeding params dicts to functions we don't own;
    for example, PyTorch optimizers. In that case we can't use ``pop_int``
    or similar to force casts (which means you can't specify ``int`` parameters
    using environment variables). This function takes something that looks JSON-like
    and recursively casts things that look like (bool, int, float) to (bool, int, float).
    """
    if isinstance(value, (int, float, bool)):
        # Already one of our desired types, so leave as is.
        return value
    elif isinstance(value, list):
        # Recursively call on each list element.
        return [infer_and_cast(item) for item in value]
    elif isinstance(value, dict):
        # Recursively call on each dict value.
        return {key: infer_and_cast(item) for key, item in value.items()}
    elif isinstance(value, str):
        # If it looks like a bool, make it a bool.
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            # See if it could be an int.
            try:
                return int(value)
            except ValueError:
                pass
            # See if it could be a float.
            try:
                return float(value)
            except ValueError:
                # Just return it as a string.
                return value
    else:
        raise ValueError(f"cannot infer type of {value}")


def _replace_none(params: Any) -> Any:
    if params == "None":
        return None
    elif isinstance(params, dict):
        for key, value in params.items():
            params[key] = _replace_none(value)
        return params
    elif isinstance(params, list):
        return [_replace_none(value) for value in params]
    return params


def _is_dict_free(obj: Any) -> bool:
    """
    Returns False if obj is a dict, or if it's a list with an element that _has_dict.
    """
    if isinstance(obj, dict):
        return False
    elif isinstance(obj, list):
        return all(_is_dict_free(item) for item in obj)
    else:
        return True


class Params(MutableMapping):

    # This allows us to check for the presence of "None" as a default argument,
    # which we require because we make a distinction between passing a value of "None"
    # and passing no value to the default parameter of "pop".
    DEFAULT = object()

    def __init__(self,
                 params: Dict[str, Any],
                 history: str = "",
                 loading_from_archive: bool = False,
                 files_to_archive: Dict[str, str] = None) -> None:
        self.params = _replace_none(params)
        self.history = history
        self.loading_from_archive = loading_from_archive
        self.files_to_archive = {} if files_to_archive is None else files_to_archive

    def add_file_to_archive(self, name: str) -> None:
        """
        Any class in its ``from_params`` method can request that some of its
        input files be added to the archive by calling this method.

        For example, if some class ``A`` had an ``input_file`` parameter, it could call

        ```
        params.add_file_to_archive("input_file")
        ```

        which would store the supplied value for ``input_file`` at the key
        ``previous.history.and.then.input_file``. The ``files_to_archive`` dict
        is shared with child instances via the ``_check_is_dict`` method, so that
        the final mapping can be retrieved from the top-level ``Params`` object.

        NOTE: You must call ``add_file_to_archive`` before you ``pop()``
        the parameter, because the ``Params`` instance looks up the value
        of the filename inside itself.

        If the ``loading_from_archive`` flag is True, this will be a no-op.
        """
        if not self.loading_from_archive:
            self.files_to_archive['{}{}'.format(self.history, name)] = self.get(name)

    def pop(self, key: str, default: Any = DEFAULT, keep_as_dict: bool = False) -> Any:
        """
        Performs the functionality associated with dict.pop(key), along with checking for
        returned dictionaries, replacing them with Param objects with an updated history
        (unless keep_as_dict is True, in which case we leave them as dictionaries).

        If ``key`` is not present in the dictionary, and no default was specified, we raise a
        ``ConfigurationError``, instead of the typical ``KeyError``.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                raise ConfigurationError("key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.params.pop(key, default)

        if keep_as_dict or _is_dict_free(value):
            logger.info(f"{self.history}{key} = {value}")
            return value
        else:
            return self._check_is_dict(key, value)

    def pop_int(self, key: str, default: Any = DEFAULT) -> int:
        """
        Performs a pop and coerces to an int.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return int(value)

    def pop_float(self, key: str, default: Any = DEFAULT) -> float:
        """
        Performs a pop and coerces to a float.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return float(value)

    def pop_bool(self, key: str, default: Any = DEFAULT) -> bool:
        """
        Performs a pop and coerces to a bool.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif value == "true":
            return True
        elif value == "false":
            return False
        else:
            raise ValueError("Cannot convert variable to bool: " + value)

    def get(self, key: str, default: Any = DEFAULT):
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.get(key)
            except KeyError:
                raise ConfigurationError('key \"{}\" is required at location \"{}\"'.format(key, self.history))
        else:
            value = self.params.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(self,
                   key: str,
                   choices: List[Any],
                   default_to_first_choice: bool = False,
                   allow_class_names: bool = True) -> Any:
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        ok_because_class_name = allow_class_names and '.' in value
        if value not in choices and not ok_because_class_name:
            key_str = self.history + key
            message = (f"{value} not in acceptable choices for {key_str}: {choices}. "
                       "You should either use the --include-package flag to make sure the correct module "
                       "is loaded, or use a fully qualified class name in your config file like "
                       """{"model": "my_module.models.MyModel"} to have it imported automatically.""")
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet: bool = False, infer_type_and_cast: bool = False):
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to PyTorch code.

        Parameters
        ----------
        quiet: bool, optional (default = False)
            Whether to log the parameters before returning them as a dict.
        infer_type_and_cast : bool, optional (default = False)
            If True, we infer types and cast (e.g. things that look like floats to floats).
        """
        if infer_type_and_cast:
            params_as_dict = infer_and_cast(self.params)
        else:
            params_as_dict = self.params

        if quiet:
            return params_as_dict

        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key  + "."
                    log_recursively(value, new_local_history)
                else:
                    logger.info(f"{history}{key} = {value}")

        logger.info("Converting Params object to dict; logging of default "
                    "values will not occur when dictionary parameters are "
                    "used subsequently.")
        logger.info("CURRENTLY DEFINED PARAMETERS: ")
        log_recursively(self.params, self.history)
        return params_as_dict

    def as_flat_dict(self):
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value

        recurse(self.params, [])
        return flat_params

    def duplicate(self) -> 'Params':
        """
        Uses ``copy.deepcopy()`` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return copy.deepcopy(self)

    def assert_empty(self, class_name: str):
        """
        Raises a ``ConfigurationError`` if ``self.params`` is not empty.  We take ``class_name`` as
        an argument so that the error message gives some idea of where an error happened, if there
        was one.  ``class_name`` should be the name of the `calling` class, the one that got extra
        parameters (if there are any).
        """
        if self.params:
            raise ConfigurationError("Extra parameters passed to {}: {}".format(class_name, self.params))

    def __getitem__(self, key):
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def _check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = self.history + new_history + "."
            return Params(value,
                          history=new_history,
                          loading_from_archive=self.loading_from_archive,
                          files_to_archive=self.files_to_archive)
        if isinstance(value, list):
            value = [self._check_is_dict(f"{new_history}.{i}", v) for i, v in enumerate(value)]
        return value

    def to_file(self, params_file: str, preference_orders: List[List[str]] = None) -> None:
        with open(params_file, "w") as handle:
            json.dump(self.as_ordered_dict(preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: List[List[str]] = None) -> OrderedDict:
        """
        Returns Ordered Dict of Params from list of partial order preferences.

        Parameters
        ----------
        preference_orders: List[List[str]], optional
            ``preference_orders`` is list of partial preference orders. ["A", "B", "C"] means
            "A" > "B" > "C". For multiple preference_orders first will be considered first.
            Keys not found, will have last but alphabetical preference. Default Preferences:
            ``[["dataset_reader", "iterator", "model", "train_data_path", "validation_data_path",
            "test_data_path", "trainer", "vocabulary"], ["type"]]``
        """
        params_dict = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = list()
            preference_orders.append(["dataset_reader", "iterator", "model",
                                      "train_data_path", "validation_data_path", "test_data_path",
                                      "trainer", "vocabulary"])
            preference_orders.append(["type"])

        def order_func(key):
            # Makes a tuple to use for ordering.  The tuple is an index into each of the `preference_orders`,
            # followed by the key itself.  This gives us integer sorting if you have a key in one of the
            # `preference_orders`, followed by alphabetical ordering if not.
            order_tuple = [order.index(key) if key in order else len(order) for order in preference_orders]
            return order_tuple + [key]

        def order_dict(dictionary, order_func):
            # Recursively orders dictionary according to scoring order_func
            result = collections.OrderedDict()
            for key, val in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
                result[key] = order_dict(val, order_func) if isinstance(val, dict) else val
            return result

        return order_dict(params_dict, order_func)

    def get_hash(self) -> str:
        """
        Returns a hash code representing the current state of this ``Params`` object.  We don't
        want to implement ``__hash__`` because that has deeper python implications (and this is a
        mutable object), but this will give you a representation of the current state.
        We use `zlib.adler32` instead of Python's builtin `hash` because the random seed for the
        latter is reset on each new program invocation, as discussed here:
        https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3.
        """
        dumped = json.dumps(self.params, sort_keys=True)
        hashed = zlib.adler32(dumped.encode())
        return str(hashed)
