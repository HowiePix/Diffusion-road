
class Registry(object):
    class_name_dict = {
        "methods": {},
        "models": {}
    }


    @classmethod
    def register_model(cls, model_cls_name):

        def wrap(model_cls):

            cls.class_name_dict["models"][model_cls_name] = model_cls

            return model_cls

        return wrap

    @classmethod
    def register_method(cls, method_cls_name):

        def wrap(method_cls):

            cls.class_name_dict["methods"][method_cls_name] = method_cls

            return method_cls

        return wrap

    @classmethod
    def get_model_class(cls, name):
        return cls.class_name_dict["models"].get(name, None)

    @classmethod
    def get_method_class(cls, name):
        return cls.class_name_dict["methods"].get(name, None)

registry = Registry()