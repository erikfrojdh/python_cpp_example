
if __name__ == '__main__':
    from distutils.core import setup, Extension
    import numpy.distutils.misc_util
    
    c_ext = Extension("mymod",
                      sources = ["src/my_module.cpp", "src/function.cpp"],
                      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
                      extra_compile_args=['-std=c++14'] )
                      
    
    c_ext.language = 'c++'
    setup(
        name= 'mymod',
        version = '1.0',
        description = 'adding two numbers',
        ext_modules=[c_ext],
    )
