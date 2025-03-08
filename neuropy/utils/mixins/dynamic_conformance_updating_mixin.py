# dynamic_conformance_updating_mixin.py
# Mixins that inherit from this mixin baseclass provide a method that allows existing instances of instantiated objects to be updated with the methods the mixin provides

class BaseDynamicInstanceConformingMixin:
    """ provides functionality to implementing mixins that allows updating  existing instances of instantiated objects to be updated with the methods the mixin provides
    
    from neuropy.utils.mixins.dynamic_conformance_updating_mixin import BaseDynamicInstanceConformingMixin
    
    
    History:
	Generalized from:
		[/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Display.py:638](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Display.py:638)
		```python
		# From `General.Pipeline.Stages.Display.PipelineWithDisplaySavingMixin`
		PipelineWithDisplaySavingMixin
		```

    """
    
    @classmethod
    def conform(cls, obj):
        """ makes the object conform to this mixin by adding all its features """
        target_class = type(obj)
        base_attrs = dir(BaseDynamicInstanceConformingMixin)
        
        # Get mixin-specific attributes
        mixin_attrs = [attr for attr in dir(cls) if attr not in base_attrs and not attr.startswith('__')]
        
        # Get class-level attributes (for class variables)
        for attr_name in mixin_attrs:
            # Handle different types of attributes
            attr = getattr(cls, attr_name)
            
            # Method handling
            if callable(attr):
                if isinstance(attr, staticmethod):
                    # Static methods
                    setattr(target_class, attr_name, staticmethod(attr.__func__))
                elif isinstance(attr, classmethod):
                    # Class methods
                    setattr(target_class, attr_name, classmethod(attr.__func__))
                else:
                    # Regular methods
                    setattr(target_class, attr_name, attr)
            
            # Class variable handling (non-callable attributes)
            else:
                setattr(target_class, attr_name, attr)
        
        # Handle class-level special attributes (like properties)
        cls_attrs = [attr for attr in dir(cls.__class__) 
                    if attr not in dir(BaseDynamicInstanceConformingMixin.__class__) 
                    and not attr.startswith('__')]
        
        for attr_name in cls_attrs:
            attr = getattr(cls.__class__, attr_name)
            
            # Property handling
            if isinstance(attr, property):
                setattr(target_class, attr_name, attr)
            
            # Other descriptor handling
            elif hasattr(attr, '__get__'):
                setattr(target_class, attr_name, attr)
                
