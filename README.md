# python_cpp_example

Minimal example building a C++ python extension. 

Useful links:
 * [Extending Python with C or C++](https://docs.python.org/3/extending/extending.html)
 * [Using NumPy C-API](https://numpy.org/doc/stable/user/c-info.html)



### Build instructions

```bash

#build in place and use from the same folder
#sometimes necessary to remove build folder and .so
#by hand
python setup.py build_ext --inplace


```

To use make sure that the .so and potentially python files are in PYTHONPATH (or installed in developer mode)

```bash
#conda
conda develop install . 

#or with pip
pip install --editable .
```

