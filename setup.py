
if __name__ == '__main__':
    from distutils.core import setup, Extension
    import numpy.distutils.misc_util
    
                      
    c_ext = Extension("_example",
                      sources = ["example.cpp", "function.cpp"], )
                      
#    c_ext.extra_compile_args = ['`root-config --cflags --glibs`']
    
    c_ext.language = 'c++'
    setup(
        name= '_example',
        version = '1.0',
        description = 'add two numbers',
        ext_modules=[c_ext],
        include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
        
    )
