import inspect
import json
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


class Tool:
    TYPE_MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    def __init__(self, func):
        if not callable(func):
            raise TypeError("Tool requires a callable")

        self.func = func
        self.name = getattr(func, "__name__", func.__class__.__name__)
        self.description = inspect.getdoc(func) or ""

        try:
            self.signature = inspect.signature(func)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot inspect signature for {self.name!r}") from exc

        # get_type_hints() can fail with unresolved forward references.
        try:
            self.hints = get_type_hints(func)
        except Exception:
            self.hints = getattr(func, "__annotations__", {}).copy()

    @property
    def schema(self):
        properties = {}
        required = []

        for name, param in self.signature.parameters.items():
            # JSON tool calls naturally map to keyword arguments.
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError(
                    f"Tool {self.name!r} contains positional-only "
                    f"parameter {name!r}, which cannot be used by this tool interface"
                )

            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                # *args cannot be represented naturally as JSON object properties.
                properties[name] = {
                    "type": "array",
                    "items": {},
                }
                continue

            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # **kwargs are represented by additional properties.
                continue

            annotation = self.hints.get(name, Any)

            schema = self._type_schema(annotation)

            if param.default is not inspect.Parameter.empty:
                # JSON-compatible defaults only.
                if self._is_json_safe(param.default):
                    schema["default"] = param.default
            else:
                required.append(name)

            properties[name] = schema

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in self.signature.parameters.values()
            ),
        }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @classmethod
    def _type_schema(cls, annotation):
        if annotation is Any:
            return {}

        if annotation is None or annotation is type(None):
            return {"type": "null"}

        # Simple types
        if annotation in cls.TYPE_MAP:
            return {"type": cls.TYPE_MAP[annotation]}

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Optional[T] / Union[T, None]
        if origin in (Union, types.UnionType):
            schemas = [cls._type_schema(arg) for arg in args]

            # Flatten a simple nullable union into anyOf.
            return {"anyOf": schemas}

        # Literal["a", "b"]
        if origin is Literal:
            values = list(args)

            schema = {
                "enum": values,
            }

            if values:
                first_type = type(values[0])
                if all(type(v) is first_type for v in values):
                    if first_type in cls.TYPE_MAP:
                        schema["type"] = cls.TYPE_MAP[first_type]

            return schema

        # list[T]
        if origin is list:
            item_schema = cls._type_schema(args[0]) if args else {}

            return {
                "type": "array",
                "items": item_schema,
            }

        # tuple[T, ...] / tuple[T, U]
        if origin is tuple:
            if not args:
                return {"type": "array"}

            # tuple[int, ...]
            if len(args) == 2 and args[1] is Ellipsis:
                return {
                    "type": "array",
                    "items": cls._type_schema(args[0]),
                }

            # tuple[int, str]
            return {
                "type": "array",
                "prefixItems": [cls._type_schema(arg) for arg in args],
            }

        # dict[K, V]
        if origin is dict:
            if len(args) == 2:
                key_type, value_type = args

                # JSON object keys are strings.
                if key_type is str:
                    return {
                        "type": "object",
                        "additionalProperties": cls._type_schema(value_type),
                    }

            return {"type": "object"}

        # Enum
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            values = [member.value for member in annotation]

            schema = {
                "enum": values,
            }

            if values:
                value_type = type(values[0])
                if value_type in cls.TYPE_MAP:
                    schema["type"] = cls.TYPE_MAP[value_type]

            return schema

        # Dataclass
        if inspect.isclass(annotation) and is_dataclass(annotation):
            properties = {}
            required = []

            for field in fields(annotation):
                field_type = field.type

                properties[field.name] = cls._type_schema(field_type)

                if field.default is field.default_factory and field.default is not None:
                    required.append(field.name)

                elif (
                    field.default is not field.default_factory
                    and field.default_factory.__class__ is not type
                ):
                    # Conservative handling; defaults are not necessarily JSON-safe.
                    pass

            return {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

        # Unknown/custom type.
        return {"type": "object"}

    def invoke(self, arguments):
        # Parse JSON arguments if necessary.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON arguments for tool {self.name!r}"
                ) from exc

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise TypeError(
                f"Arguments for tool {self.name!r} must be an object/dict, "
                f"got {type(arguments).__name__}"
            )

        self._validate_arguments(arguments)

        return self.func(**arguments)

    def _validate_arguments(self, arguments):
        parameters = self.signature.parameters

        # Reject unknown arguments unless **kwargs exists.
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )

        if not accepts_kwargs:
            unknown = set(arguments) - set(parameters)

            if unknown:
                raise TypeError(
                    f"Unexpected argument(s) for {self.name!r}: "
                    f"{', '.join(sorted(unknown))}"
                )

        # Validate supplied arguments.
        for name, value in arguments.items():
            param = parameters.get(name)

            # Belongs to **kwargs.
            if param is None:
                continue

            annotation = self.hints.get(name, Any)

            self._validate_type(
                value,
                annotation,
                path=name,
            )

    @classmethod
    def _validate_type(cls, value, annotation, path="value"):
        if annotation is Any:
            return

        if annotation is None or annotation is type(None):
            if value is not None:
                raise TypeError(f"{path} must be None, got {type(value).__name__}")
            return

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Union / Optional
        if origin in (Union, types.UnionType):
            for arg in args:
                try:
                    cls._validate_type(value, arg, path)
                    return
                except TypeError:
                    pass

            raise TypeError(f"{path} has invalid type/value: {value!r}")

        # Literal
        if origin is Literal:
            if value not in args:
                raise TypeError(f"{path} must be one of {args}, got {value!r}")
            return

        # bool must be checked before int because bool subclasses int.
        if annotation is bool:
            if not isinstance(value, bool):
                raise TypeError(f"{path} must be bool, got {type(value).__name__}")
            return

        if annotation is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{path} must be int, got {type(value).__name__}")
            return

        if annotation is float:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{path} must be float, got {type(value).__name__}")
            return

        if annotation is str:
            if not isinstance(value, str):
                raise TypeError(f"{path} must be str, got {type(value).__name__}")
            return

        # list[T]
        if origin is list:
            if not isinstance(value, list):
                raise TypeError(f"{path} must be list, got {type(value).__name__}")

            if args:
                for i, item in enumerate(value):
                    cls._validate_type(
                        item,
                        args[0],
                        f"{path}[{i}]",
                    )
            return

        # tuple
        if origin is tuple:
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"{path} must be array/tuple, got {type(value).__name__}"
                )

            if len(args) == 2 and args[1] is Ellipsis:
                for i, item in enumerate(value):
                    cls._validate_type(
                        item,
                        args[0],
                        f"{path}[{i}]",
                    )

            elif args:
                if len(value) != len(args):
                    raise TypeError(f"{path} must contain {len(args)} items")

                for i, (item, expected) in enumerate(zip(value, args)):
                    cls._validate_type(
                        item,
                        expected,
                        f"{path}[{i}]",
                    )

            return

        # dict[K, V]
        if origin is dict:
            if not isinstance(value, dict):
                raise TypeError(
                    f"{path} must be object/dict, got {type(value).__name__}"
                )

            if len(args) == 2:
                key_type, value_type = args

                for key, item in value.items():
                    cls._validate_type(
                        key,
                        key_type,
                        f"{path} key",
                    )
                    cls._validate_type(
                        item,
                        value_type,
                        f"{path}[{key!r}]",
                    )
            return

        # Enum
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            valid_values = [member.value for member in annotation]

            if value not in valid_values:
                raise TypeError(f"{path} must be one of {valid_values}, got {value!r}")
            return

        # Dataclass
        if inspect.isclass(annotation) and is_dataclass(annotation):
            if not isinstance(value, dict):
                raise TypeError(f"{path} must be object/dict")

            try:
                hints = get_type_hints(annotation)
            except Exception:
                hints = getattr(annotation, "__annotations__", {})

            for field in fields(annotation):
                if field.name not in value:
                    if (
                        field.default is field.default_factory
                        and field.default is not None
                    ):
                        raise TypeError(f"{path}.{field.name} is required")
                    continue

                cls._validate_type(
                    value[field.name],
                    hints.get(field.name, field.type),
                    f"{path}.{field.name}",
                )

            return

        # Custom classes.
        if inspect.isclass(annotation):
            if not isinstance(value, annotation):
                raise TypeError(
                    f"{path} must be {annotation.__name__}, got {type(value).__name__}"
                )

    @staticmethod
    def _is_json_safe(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)



if __name__ == "__main__":

    @Tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.
    
        Args:
            a: The first number.
            b: The second number.
    
        Returns:
            The product of a and b.
        """
        return a * b
    
    
    @Tool
    def get_weather(loc: str) -> str:
        """Get the weather for a location.
    
        Args:
            loc: The name of the location to get the weather for.
    
        Returns:
            The current temperature at the location.
        """
        return f"its 10 degrees Celsius in {loc}"

    tools = [multiply, get_weather]
    print(tool_schema = [tool.schema for tool in tools])