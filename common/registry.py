
class Registry(object):
    class_name_dict = {
        "methods": {},
        "models": {},
        "evaluators": {},
        "processors": {}
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
    def register_evaluator(cls, evaluator_cls_name):

        def wrap(evaluator_cls):

            cls.class_name_dict["evaluators"][evaluator_cls_name] = evaluator_cls

            return evaluator_cls

        return wrap

    @classmethod
    def register_processor(cls, processor_cls_name):

        def wrap(processor_cls):

            cls.class_name_dict["processors"][processor_cls_name] = processor_cls

            return processor_cls

        return wrap

    @classmethod
    def get_model_class(cls, name):
        return cls.class_name_dict["models"].get(name, None)

    @classmethod
    def get_method_class(cls, name):
        return cls.class_name_dict["methods"].get(name, None)
    
    @classmethod
    def get_evaluator_class(cls, name):
        return cls.class_name_dict["evaluators"].get(name, None)
    
    @classmethod
    def get_processor_class(cls, name):
        return cls.class_name_dict["processors"].get(name, None)

registry = Registry()