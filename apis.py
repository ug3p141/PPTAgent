import inspect

from presentation import SlidePage

slide = None


class ModelAPI:
    def __init__(self, local_vars: dict):
        self.registered_functions = {}
        for name, obj in local_vars.items():
            if inspect.isfunction(obj):
                self.registered_functions[name] = obj

    def get_apis_docs(self):
        return "\n".join(
            [self._api_doc(name) for name in self.registered_functions.keys()]
        )

    def _api_doc(self, api_name: str):
        assert (
            api_name in self.registered_functions
        ), f"{api_name} is not a registered function."
        func = self.registered_functions[api_name]
        sig = inspect.signature(func)
        params = []
        for name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                params.append(f"{name}: {param.annotation.__name__}")
            else:
                params.append(f"{name}")
        signature = f"def {func.__name__}({', '.join(params)})"
        doc = inspect.getdoc(func)
        return f"{signature}\n{doc}"

    def execute_model_output(self, exec_slide: SlidePage, model_output: str):
        global slide
        slide = exec_slide
        lines = model_output.strip().split("\n")
        results = []
        # 可以把东西放进这个locals里
        for line in lines:
            result = eval(line, {"slide": slide}, self.registered_functions)
            results.append(result)
        return results


def set_text(text: str, id: str):
    """
    This function sets the text of the element with the given id.
    """
    global slide


def set_image(image_path: str, image_id: str):
    """
    This function sets the image of the element with the given id.
    """
    global slide


def clone_shape(shape_id: str, shape_bounds: dict):
    """
    This function clones the shape with the given id, applying the specified bounding box to the new shape.

    Args:
        shape_id (str): The unique identifier of the shape to clone.
        shape_bounds (dict): A dictionary containing the bounding box parameters of the shape, including `left`, `top`, `width`, and `height`.
        eg. `{'left': 100, 'top': 200, 'width': 300, 'height': 400}`
    """
    global slide


def del_shape(shape_id: str):
    """
    This function deletes the shape with the given id.
    """
    pass


def swap_style(source_shape_id: str, target_shape_id: str):
    """
    This function swaps the style of the source shape with the style of the target shape.
    In certain scenarios, one shape in a list may need to become a focal element (e.g., bold, highlighted with different colors, underlined).
    During setting elements, it is sometimes necessary to modify the focal element based on its content.
    """
    pass


# def

model_api = ModelAPI(locals())
if __name__ == "__main__":

    def add(x: int, y: int, **kwargs) -> int:
        """
        This function adds two integers and returns the result.
        """
        global slide
        slide = x + y

    model_api = ModelAPI(locals())
    print(model_api.get_apis_docs())
    model_api.execute_model_output(2, "add(1,2)")
    assert slide == 3
