

import setuptools

c_ext = setuptools.Extension("mymod",
                    sources = ["src/my_module.cpp", "src/function.cpp"],
                    extra_compile_args=['-std=c++11', '-Wall'] )
                    

c_ext.language = 'c++'
setuptools.setup(
    name= 'mymod',
    version = '1.0',
    description = 'adding two numbers',
    ext_modules=[c_ext],
)
