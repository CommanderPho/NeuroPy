# dynamic_conformance_updating_mixin.py
# Mixins that inherit from this mixin baseclass provide a method that allows existing instances of instantiated objects to be updated with the methods the mixin provides

class BaseDynamicInstanceConformingMixin:
    """ provides functionality to implementing mixins that allows updating  existing instances of instantiated objects to be updated with the methods the mixin provides
    
    from neuropy.utils.mixins.dynamic_conformance_updating_mixin import BaseDynamicInstanceConformingMixin
    
    """
    
    @classmethod
    def conform(cls, obj):
        """ makes the object conform to this mixin by adding its methods and properties """
        target_class = type(obj)
        
        # For methods
        def conform_to_implementing_method(func):
            setattr(target_class, func.__name__, func)
        
        # For properties
        def conform_to_implementing_property(prop_name):
            prop_obj = getattr(cls.__class__, prop_name)
            setattr(target_class, prop_name, prop_obj)
        
        # Get all attributes from the mixin class
        base_attrs = dir(BaseDynamicInstanceConformingMixin)
        
        # Handle methods
        mixin_methods = [method for method in dir(cls) 
                         if callable(getattr(cls, method)) 
                         and method not in base_attrs 
                         and not method.startswith('__')]
        
        for method_name in mixin_methods:
            method = getattr(cls, method_name)
            conform_to_implementing_method(method)
        
        # Handle properties
        for attr_name in dir(cls.__class__):
            if attr_name not in base_attrs and not attr_name.startswith('__'):
                attr = getattr(cls.__class__, attr_name)
                if isinstance(attr, property):
                    conform_to_implementing_property(attr_name)
